"""
bench_tiempos.py -- perfilado algoritmico AISLADO de la E/S serial (Sec. 4.5 del manuscrito)
=========================================================================================
Mide, en el MISMO PC y entorno donde lo ejecutes, el tiempo por paso de:
  1. Observador Koopman (igual que el lazo en vivo: z+ = A z + B u + L (y - C z), recortes z>=0)
  2. EKF completo (predict + update) con el Jacobiano de DIFERENCIAS FINITAS usado en vivo
       - desglose: Jacobiano FD, RK4, update
  3. Jacobiano con diferenciacion algoritmica (CasADi), llamado desde Python (incluye overhead de llamada)
  4. MPC (IPOPT via casadi.nlpsol, N=20, q=500, wk=10 en el ultimo paso, warm-start desplazado),
     en un lazo cerrado SIMULADO (planta = ODE con RK4 + ruido de medicion) con el observador Koopman
  5. Lazo completo observador + MPC

No necesita CSV ni el ESP32.  Uso:   python bench_tiempos.py
Salida: consola + results/bench_tiempos.txt + results/bench_tiempos.csv
Copia el bloque "ENTORNO" a la seccion de hardware del articulo.
"""
import os, sys, time, platform, subprocess, warnings
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")

N_FAST = 5000      # repeticiones para operaciones rapidas
N_LOOP = 400       # pasos del lazo cerrado simulado (= 800 s)
WARMUP = 50
RNG = np.random.default_rng(0)

P = {'Atank': 3.02e-3, 'rho': 997.0, 'eta': 8.9e-4, 'g': 9.81, 'alphao1': 0.0271, 'Do1': 10e-3, 'alphao2': 0.0271,
     'A2': 7.85e-5, 'alphao3': 0.0271, 'Do3': 10e-3, 'alpha120': 0.3038, 'D12': 10e-3, 'A12': 7.85e-5, 'lambdac12': 24000,
     'alpha230': 0.1344, 'D23': 10e-3, 'A23': 7.85e-5, 'lambdac23': 29600, 'qi1max': 2.69e-5, 'qi3max': 2.41e-5}
Ts, EPS, EPS_PSI = 2.0, 1e-10, 1e-8
EKF_Q = np.diag([(0.5e-3) ** 2, (2.0e-3) ** 2, (0.5e-3) ** 2]); EKF_R = np.diag([(1.5e-3) ** 2] * 2)
EKF_P0 = np.diag([(5e-3) ** 2, (20e-3) ** 2, (5e-3) ** 2])


# ------------------------------------------------------------------ modelo (copia del codigo en vivo)
def nl_ode(x, u, p=P):
    h1 = max(x[0], EPS); h2 = max(x[1], EPS); h3 = max(x[2], EPS)
    qi1 = p['qi1max'] * np.clip(u[0], 0, 1); qi3 = p['qi3max'] * np.clip(u[1], 0, 1)
    qo1 = p['alphao1'] * (p['Do1'] ** 2 * np.pi / 4) * np.sqrt(2 * p['g'] * h1)
    qo2 = p['alphao2'] * p['A2'] * np.sqrt(2 * p['g'] * h2)
    qo3 = p['alphao3'] * (p['Do3'] ** 2 * np.pi / 4) * np.sqrt(2 * p['g'] * h3)
    d12 = h1 - h2; l12 = p['D12'] * p['rho'] / p['eta'] * np.sqrt(2 * p['g'] * abs(d12) + EPS)
    q12 = p['alpha120'] * np.tanh(2 * l12 / p['lambdac12']) * p['A12'] * np.sqrt(2 * p['g'] * abs(d12) + EPS) * np.sign(d12 + 1e-15)
    d23 = h2 - h3; l23 = p['D23'] * p['rho'] / p['eta'] * np.sqrt(2 * p['g'] * abs(d23) + EPS)
    q23 = p['alpha230'] * np.tanh(2 * l23 / p['lambdac23']) * p['A23'] * np.sqrt(2 * p['g'] * abs(d23) + EPS) * np.sign(d23 + 1e-15)
    return np.array([(qi1 - q12 - qo1) / p['Atank'], (q12 - q23 - qo2) / p['Atank'], (qi3 + q23 - qo3) / p['Atank']])


def rk4(x, u, dt=Ts):
    k1 = nl_ode(x, u); k2 = nl_ode(x + dt / 2 * k1, u); k3 = nl_ode(x + dt / 2 * k2, u); k4 = nl_ode(x + dt * k3, u)
    return np.clip(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0, 0.27)


def psi(x):
    h1, h2, h3 = x
    return np.array([np.sqrt(max(h1, EPS_PSI)), np.sqrt(max(h2, EPS_PSI)), np.sqrt(max(h3, EPS_PSI)), h1 * h2, h2 * h3, h1 * h3, h1, h3])


def compute_ss(h2_ref):
    def eqs(v):
        h1, h3, u1, u3 = v
        return np.concatenate([nl_ode([max(h1, 1e-4), h2_ref, max(h3, 1e-4)], [np.clip(u1, 0, 1), np.clip(u3, 0, 1)]), [0.05 * v[3]]])
    gs = [(1.4, .8, .5, 0), (1.6, .7, .7, 0), (2.0, .8, .9, 0), (2.5, .8, .7, 0), (2.5, .8, .75, 0), (1.8, .85, .85, 0),
          (1.3, .9, 1, .1), (1.2, 1, 1, .3), (1.1, 1.1, 1, .5), (1.0, 1.2, 1, .8)]
    best, br = None, 1e10
    for a, b, c, d in gs:
        try:
            s = least_squares(eqs, [h2_ref * a, h2_ref * b, c, d], bounds=([.01, .01, 0, 0], [.27, .27, 1, 1]), max_nfev=5000)
            r = np.linalg.norm(s.fun[:3])
            if s.success and r < br: br, best = r, s.x
        except Exception:
            pass
    if best is not None and br < 1e-4:
        return np.array([best[0], h2_ref, best[1]]), np.array([np.clip(best[2], 0, 1), np.clip(best[3], 0, 1)])
    return None, None


def train_edmd():
    rng = np.random.default_rng(42); Zk, Uk, Zn = [], [], []
    def roll(x, h2t, gain, noise, steps, u_ss, pert_u3=True):
        for _ in range(steps):
            err = x[1] - h2t; u = u_ss.copy()
            u[0] += np.clip(-gain * err, -gain, gain) + rng.uniform(-noise, noise)
            if pert_u3: u[1] += rng.uniform(-0.05, 0.05)
            u = np.clip(u, 0, 1); xn = rk4(x, u)
            Zk.append(psi(x)); Uk.append(u); Zn.append(psi(xn)); x = xn
    for h2r in [.10, .12, .15, .18, .20, .22]:
        xs, us = compute_ss(h2r)
        if xs is None: continue
        for pert in [.95, .90, .98, .85, .92, .88]:
            x = xs.copy(); x[1] = np.clip(h2r * pert, .05, .7); roll(x, h2r, .35, .15, 170, us)
    for h2r in [.10, .15, .20]:
        xs, us = compute_ss(h2r)
        if xs is None: continue
        for pert in [.60, .70, .75, 1.25, 1.35]:
            x = xs.copy(); x[1] = np.clip(h2r * pert, .05, .7); roll(x, h2r, .6, .1, 100, us, pert_u3=False)
    for h2f, h2t in [(.10, .20), (.20, .15), (.15, .10)]:
        xf, uf = compute_ss(h2f); xt, _ = compute_ss(h2t)
        if xf is None or xt is None: continue
        roll(xf.copy(), h2t, .5, .08, 200, uf, pert_u3=False)
    Z, U, Zn_ = np.array(Zk), np.array(Uk), np.array(Zn); nz, nu = Z.shape[1], U.shape[1]
    Phi = np.hstack([Z, U]); Th = np.linalg.solve(Phi.T @ Phi + 1e-4 * np.eye(nz + nu), Phi.T @ Zn_)
    Ar, B = Th[:nz].T, Th[nz:].T
    v, V = np.linalg.eig(Ar); vs = np.where(np.abs(v) >= 1, v / np.abs(v) * (1 - 1e-4), v)
    return np.real(V @ np.diag(vs) @ np.linalg.inv(V)), B


def design(A, alpha=1.0):
    C = np.zeros((4, 8)); C[0, 0] = C[1, 2] = C[2, 6] = C[3, 7] = 1.0
    X = la.solve_discrete_are(A.T, C.T, alpha * np.eye(8), np.eye(4))
    return np.linalg.solve(C @ X @ C.T + np.eye(4), C @ X @ A.T).T, C


# ------------------------------------------------------------------ EKF (igual que ekf_koopman_real.py)
def jac_fd(x, u, eps=1e-6):
    f0 = nl_ode(x, u); J = np.zeros((3, 3))
    for i in range(3):
        xp = x.copy(); xp[i] += eps; J[:, i] = (nl_ode(xp, u) - f0) / eps
    return np.eye(3) + Ts * J


class EKF:
    Cm = np.array([[1., 0, 0], [0, 0, 1.]])
    def __init__(s, x0): s.x = np.array(x0, float); s.P = EKF_P0.copy()
    def predict(s, u):
        F = jac_fd(s.x, u); s.x = np.clip(rk4(s.x, u), 0, .75); s.P = F @ s.P @ F.T + EKF_Q
    def update(s, y):
        S = s.Cm @ s.P @ s.Cm.T + EKF_R; K = s.P @ s.Cm.T @ np.linalg.inv(S)
        s.x = np.clip(s.x + K @ (y - s.Cm @ s.x), 0, .75); s.P = (np.eye(3) - K @ s.Cm) @ s.P; s.P = .5 * (s.P + s.P.T)
        return s.x.copy()


def koopman_step(z, u, y, A, B, L, C):
    zp = A @ z + B @ u; zp[:3] = np.maximum(zp[:3], 0); zp[6:] = np.maximum(zp[6:], 0)
    z = zp + L @ (y - C @ zp); z[:3] = np.maximum(z[:3], 0); z[6:] = np.maximum(z[6:], 0)
    return z


# ------------------------------------------------------------------ utilidades
def stats_ms(ns):
    a = np.asarray(ns[WARMUP:], float) / 1e6
    return dict(mean=a.mean(), std=a.std(ddof=1), p99=np.percentile(a, 99), n=len(a))


def timeit(fn, n):
    out = []
    for _ in range(n + WARMUP):
        t0 = time.perf_counter_ns(); fn(); out.append(time.perf_counter_ns() - t0)
    return stats_ms(out)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
    except Exception:
        return "n/d"


def env_info():
    info = {"Python": sys.version.split()[0], "Ejecutable": sys.executable, "SO": platform.platform(),
            "numpy": np.__version__, "scipy": la.__name__ and __import__("scipy").__version__}
    if os.name == "nt":
        info["CPU"] = sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"')
        info["Nucleos/hilos"] = sh('powershell -NoProfile -Command "$p=Get-CimInstance Win32_Processor;\\"$($p.NumberOfCores)/$($p.NumberOfLogicalProcessors)\\""')
        info["RAM [GB]"] = sh('powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"')
        info["Plan de energia"] = sh('powercfg /getactivescheme')
    else:
        info["CPU"] = platform.processor() or sh("lscpu | grep 'Model name' | cut -d: -f2")
    try:
        import casadi; info["casadi"] = casadi.__version__
    except Exception:
        info["casadi"] = "no instalado"
    return info


# ------------------------------------------------------------------ MPC (igual que build_mpc de los scripts en vivo)
def build_mpc(A, B):
    import casadi as ca
    nz, nu, N, q = 8, 2, 20, 500
    Z = ca.SX.sym('Z', nz, N + 1); U = ca.SX.sym('U', nu, N); z0 = ca.SX.sym('z0', nz); yr = ca.SX.sym('yr'); up = ca.SX.sym('up', nu)
    Ac, Bc = ca.DM(A), ca.DM(B); J = 0; g = [Z[:, 0] - z0]; lbg = [0.] * nz; ubg = [0.] * nz
    R1, R2 = ca.DM(np.diag([.1, 1.])), ca.DM(np.diag([.1, .1]))
    for k in range(N):
        g += [Z[:, k + 1] - Ac @ Z[:, k] - Bc @ U[:, k]]; lbg += [0.] * nz; ubg += [0.] * nz
        ey = Z[1, k + 1] ** 2 - yr; du = U[:, 0] - up if k == 0 else U[:, k] - U[:, k - 1]; wk = 10 if k == N - 1 else 1
        J += wk * q * ey ** 2 + ca.mtimes(U[:, k].T, R1 @ U[:, k]) + ca.mtimes(du.T, R2 @ du)
    w = ca.vertcat(ca.reshape(Z, nz * (N + 1), 1), ca.reshape(U, nu * N, 1)); par = ca.vertcat(z0, yr, up)
    lbw = [-ca.inf] * (nz * (N + 1)) + [0.] * (nu * N); ubw = [ca.inf] * (nz * (N + 1)) + [1.] * (nu * N)
    opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 500, 'ipopt.tol': 1e-5, 'ipopt.acceptable_tol': 1e-4, 'print_time': 0}
    return ca.nlpsol('s', 'ipopt', {'x': w, 'f': J, 'g': ca.vertcat(*g), 'p': par}, opts), N, nz, nu, lbw, ubw, lbg, ubg


def jac_ad_function():
    import casadi as ca
    xs, us = ca.SX.sym('x', 3), ca.SX.sym('u', 2); p = P; h = [ca.fmax(xs[i], EPS) for i in range(3)]
    qi1 = p['qi1max'] * ca.fmin(ca.fmax(us[0], 0), 1); qi3 = p['qi3max'] * ca.fmin(ca.fmax(us[1], 0), 1)
    qo = [p['alphao1'] * (p['Do1'] ** 2 * np.pi / 4) * ca.sqrt(2 * p['g'] * h[0]), p['alphao2'] * p['A2'] * ca.sqrt(2 * p['g'] * h[1]),
          p['alphao3'] * (p['Do3'] ** 2 * np.pi / 4) * ca.sqrt(2 * p['g'] * h[2])]
    def cpl(da, a0, D, Ar, lc):
        s = ca.sqrt(2 * p['g'] * ca.fabs(da) + EPS)
        return p[a0] * ca.tanh(2 * (p[D] * p['rho'] / p['eta'] * s) / p[lc]) * p[Ar] * s * ca.sign(da + 1e-15)
    q12 = cpl(h[0] - h[1], 'alpha120', 'D12', 'A12', 'lambdac12'); q23 = cpl(h[1] - h[2], 'alpha230', 'D23', 'A23', 'lambdac23')
    f = ca.vertcat((qi1 - q12 - qo[0]) / p['Atank'], (q12 - q23 - qo[1]) / p['Atank'], (qi3 + q23 - qo[2]) / p['Atank'])
    return ca.Function('Fk', [xs, us], [ca.SX.eye(3) + Ts * ca.jacobian(f, xs)])


def main():
    os.makedirs("results", exist_ok=True)
    rows = []; log = []
    def out(s=""):
        print(s); log.append(s)
    info = env_info()
    out("=" * 78); out(" ENTORNO"); out("=" * 78)
    for k, v in info.items(): out(f"  {k:<16}: {v}")
    out("  (cierra navegador/otros programas y conecta el cargador; usa el plan de energia 'Alto rendimiento' para medir)")

    out("\nEntrenando EDMD y disenando L (alpha=1, como en los experimentos)...")
    A, B = train_edmd(); L, C = design(A); rho = np.abs(np.linalg.eigvals(A - L @ C)).max()
    out(f"  rho(A)={np.abs(np.linalg.eigvals(A)).max():.5f} | rho(A-LC)={rho:.5f} | |L|max={np.abs(L).max():.4f}")

    x = np.array([.15, .12, .10]); u = np.array([.6, .1]); z = psi(x); y4 = np.array([np.sqrt(.15), np.sqrt(.10), .15, .10])
    ekf = EKF(x)
    def rec(name, s):
        rows.append((name, s['mean'], s['std'], s['p99'], s['n'])); out(f"  {name:<52}{s['mean']:>10.4f}{s['std']:>10.4f}{s['p99']:>10.4f}   (n={s['n']})")

    out("\n" + "=" * 78); out(f" TIEMPOS POR PASO [ms]  {'media':>21}{'std':>10}{'P99':>10}"); out("=" * 78)
    rec("Observador Koopman (A z + B u + L(y-Cz))", timeit(lambda: koopman_step(z.copy(), u, y4, A, B, L, C), N_FAST))
    rec("EKF predict+update (Jacobiano FD, como en vivo)", timeit(lambda: (ekf.predict(u), ekf.update(np.array([.15, .10]))), N_FAST))
    rec("  - Jacobiano por diferencias finitas (4 evals ODE)", timeit(lambda: jac_fd(x, u), N_FAST))
    rec("  - RK4 (4 evals ODE)", timeit(lambda: rk4(x, u), N_FAST))
    try:
        Fk = jac_ad_function(); Fk(x, u)
        rec("Jacobiano CasADi AD (llamada desde Python)", timeit(lambda: Fk(x, u), N_FAST))
        d = np.abs(np.array(Fk(x, u)) - jac_fd(x, u)).max()
        out(f"  max|J_AD - J_FD| = {d:.2e}   (con eps=1e-6 hacia adelante no esperes < 1e-7)")
    except Exception as e:
        out(f"  CasADi AD omitido: {e}")

    try:
        solver, N, nz, nu, lbw, ubw, lbg, ubg = build_mpc(A, B)
        SS = {r: compute_ss(r) for r in (.10, .15, .20)}
        refs = np.where(np.arange(N_LOOP) < 132, .10, np.where(np.arange(N_LOOP) < 264, .20, .15))
        xp = np.array([.02, .02, .02]); zo = psi(np.array([xp[0], refs[0], xp[2]])); uprev = np.zeros(2); w0 = None; yprev = refs[0]
        t_obs, t_mpc, iters, fails = [], [], [], 0
        for k in range(N_LOOP):
            yr = refs[k]
            if yr != yprev:
                xs, us = SS[yr]
                w0 = np.concatenate([np.tile(psi(xs), (N + 1, 1)).ravel(), np.tile(us, (N, 1)).ravel()]) if xs is not None else None
            yprev = yr
            xp = rk4(xp, uprev); ym = xp[[0, 2]] + RNG.normal(0, 3e-3, 2)
            y4 = np.array([np.sqrt(max(ym[0], EPS_PSI)), np.sqrt(max(ym[1], EPS_PSI)), ym[0], ym[1]])
            t0 = time.perf_counter_ns(); zo = koopman_step(zo, uprev, y4, A, B, L, C); t1 = time.perf_counter_ns()
            if w0 is None: w0 = np.zeros(nz * (N + 1) + nu * N); w0[:nz] = zo
            t2 = time.perf_counter_ns()
            try:
                sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=np.concatenate([zo, [yr], uprev])); t3 = time.perf_counter_ns()
                wo = np.array(sol['x']).ravel(); Zo = wo[:nz * (N + 1)].reshape(N + 1, nz); Uo = wo[nz * (N + 1):].reshape(N, nu)
                uprev = np.clip(Uo[0], 0, 1); w0 = np.concatenate([np.vstack([Zo[1:], Zo[-1:]]).ravel(), np.vstack([Uo[1:], Uo[-1:]]).ravel()])
                iters.append(solver.stats().get('iter_count', np.nan))
            except Exception:
                fails += 1; w0 = None; t3 = time.perf_counter_ns()
            t_obs.append(t1 - t0); t_mpc.append(t3 - t2)
        rec("MPC IPOPT N=20 (lazo cerrado simulado, warm-start)", stats_ms(t_mpc))
        rec("Lazo completo: observador + MPC", stats_ms([a + b for a, b in zip(t_obs, t_mpc)]))
        out(f"  iteraciones IPOPT: media {np.nanmean(iters):.1f} | fallos {fails}")
    except Exception as e:
        out(f"  MPC omitido: {e}")

    out("\nNota: estos tiempos son de calculo puro. El periodo real del lazo (~2.1 s) lo fija el firmware del ESP32, no el PC.")
    open("results/bench_tiempos.txt", "w", encoding="utf-8").write("\n".join(log))
    import csv
    with open("results/bench_tiempos.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["componente", "media_ms", "std_ms", "p99_ms", "n"]); w.writerows(rows)
    out("Guardado: results/bench_tiempos.txt y results/bench_tiempos.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        N_FAST, N_LOOP = 300, 60
    main()