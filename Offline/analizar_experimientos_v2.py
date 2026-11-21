import os, sys, re, glob, time
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy import stats
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARÁMETROS GLOBALES
# ============================================================================
P_NOM = {
    'Atank'   : 3.02e-3, 'rho': 997.0, 'eta': 8.9e-4, 'g': 9.81,
    'alphao1' : 0.0271, 'Do1': 10e-3, 'alphao2': 0.0271, 'A2': 7.85e-5, 'alphao3': 0.0271, 'Do3': 10e-3,
    'alpha120': 0.3038, 'D12': 10e-3, 'A12': 7.85e-5, 'lambdac12': 24000,
    'alpha230': 0.1344, 'D23': 10e-3, 'A23': 7.85e-5, 'lambdac23': 29600,
    'qi1max'  : 2.69e-5, 'qi3max': 2.41e-5
}
Ts = 2.0; T_sim = 800; Nsim = int(T_sim / Ts)
EPS = 1e-10; EPS_PSI = 1e-8; SETTLING_S = 60
SEGS = [(0, 264, 0.10), (264, 528, 0.20), (528, 800, 0.15)]
EKF_Q = np.diag([(0.5e-3)**2, (2.0e-3)**2, (0.5e-3)**2])
EKF_R = np.diag([(1.5e-3)**2, (1.5e-3)**2])
EKF_P0 = np.diag([(5e-3)**2, (20e-3)**2, (5e-3)**2])
MAX_DELTA_H = 0.05

# **Ruta que me pediste**
CSV_DIR_DEFAULT = r"C:\Users\PC\Downloads\articulo"
_T_OUT = 'salida_tuning'

# ============================================================================
# FUNCIONES BASE (ODE, EDMD, OBSERVER)
# ============================================================================
def nl_ode(x, u, p=P_NOM):
    h1 = max(x[0], EPS); h2 = max(x[1], EPS); h3 = max(x[2], EPS)
    qi1 = p['qi1max'] * np.clip(u[0], 0, 1); qi3 = p['qi3max'] * np.clip(u[1], 0, 1)
    qo1 = p['alphao1'] * (p['Do1']**2 * np.pi / 4) * np.sqrt(2 * p['g'] * h1)
    qo2 = p['alphao2'] * p['A2'] * np.sqrt(2 * p['g'] * h2)
    qo3 = p['alphao3'] * (p['Do3']**2 * np.pi / 4) * np.sqrt(2 * p['g'] * h3)
    dh12 = h1 - h2; l12 = p['D12'] * p['rho'] / p['eta'] * np.sqrt(2 * p['g'] * abs(dh12) + EPS)
    a12 = p['alpha120'] * np.tanh(2 * l12 / p['lambdac12'])
    q12 = a12 * p['A12'] * np.sqrt(2 * p['g'] * abs(dh12) + EPS) * np.sign(dh12 + 1e-15)
    dh23 = h2 - h3; l23 = p['D23'] * p['rho'] / p['eta'] * np.sqrt(2 * p['g'] * abs(dh23) + EPS)
    a23 = p['alpha230'] * np.tanh(2 * l23 / p['lambdac23'])
    q23 = a23 * p['A23'] * np.sqrt(2 * p['g'] * abs(dh23) + EPS) * np.sign(dh23 + 1e-15)
    return np.array([(qi1 - q12 - qo1) / p['Atank'], (q12 - q23 - qo2) / p['Atank'], (qi3 + q23 - qo3) / p['Atank']])

def rk4(x, u, p=P_NOM, dt=Ts):
    k1 = nl_ode(x, u, p); k2 = nl_ode(x + dt/2*k1, u, p)
    k3 = nl_ode(x + dt/2*k2, u, p); k4 = nl_ode(x + dt*k3, u, p)
    return np.clip(x + dt/6*(k1 + 2*k2 + 2*k3 + k4), 0, 0.27)

def psi(x):
    h1, h2, h3 = x[0], x[1], x[2]
    return np.array([np.sqrt(max(h1, EPS_PSI)), np.sqrt(max(h2, EPS_PSI)), np.sqrt(max(h3, EPS_PSI)),
                     h1 * h2, h2 * h3, h1 * h3, h1, h3], dtype=float)

def compute_ss(h2_ref, p=P_NOM):
    w_reg = 0.05
    def eqs(v):
        h1, h3, u1, u3 = v
        f = nl_ode([max(h1, 1e-4), h2_ref, max(h3, 1e-4)], [np.clip(u1, 0, 1), np.clip(u3, 0, 1)], p)
        return np.concatenate([f, [w_reg * u3]])
    guesses = [(h2_ref*1.4, h2_ref*0.8, 0.5, 0.0), (h2_ref*1.6, h2_ref*0.7, 0.7, 0.0),
               (h2_ref*2.0, h2_ref*0.8, 0.9, 0.0), (h2_ref*2.5, h2_ref*0.8, 0.7, 0.0),
               (h2_ref*2.5, h2_ref*0.8, 0.75, 0.0), (h2_ref*1.8, h2_ref*0.85, 0.85, 0.0),
               (h2_ref*1.3, h2_ref*0.9, 1.0, 0.1), (h2_ref*1.2, h2_ref*1.0, 1.0, 0.3),
               (h2_ref*1.1, h2_ref*1.1, 1.0, 0.5), (h2_ref*1.0, h2_ref*1.2, 1.0, 0.8)]
    best, best_res = None, 1e10
    for g in guesses:
        try:
            s = least_squares(eqs, list(g), bounds=([0.01, 0.01, 0.0, 0.0], [0.27, 0.27, 1.0, 1.0]), max_nfev=5000)
            phys_res = np.linalg.norm(s.fun[:3])
            if s.success and phys_res < best_res: best_res = phys_res; best = s.x
        except: pass
    if best is not None and best_res < 1e-4: return (np.array([best[0], h2_ref, best[1]]), np.array([np.clip(best[2], 0, 1), np.clip(best[3], 0, 1)]))
    return None, None

def train_edmd(verbose=False):
    rng = np.random.default_rng(42); z_k, u_k, z_k1 = [], [], []
    for h2r in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22]:
        x_ss, u_ss = compute_ss(h2r)
        if x_ss is None: continue
        for pert in [0.95, 0.90, 0.98, 0.85, 0.92, 0.88]:
            x = x_ss.copy(); x[1] = np.clip(h2r * pert, 0.05, 0.70)
            for _ in range(170):
                err = x[1] - h2r; u = u_ss.copy()
                u[0] += np.clip(-0.35*err, -0.35, 0.35) + rng.uniform(-0.15, 0.15); u[1] += rng.uniform(-0.05, 0.05); u = np.clip(u, 0, 1)
                z = psi(x); xn = rk4(x, u); zn = psi(xn)
                z_k.append(z); u_k.append(u); z_k1.append(zn); x = xn
    for h2r in [0.10, 0.15, 0.20]:
        x_ss, u_ss = compute_ss(h2r)
        if x_ss is None: continue
        for pert in [0.60, 0.70, 0.75, 1.25, 1.35]:
            x = x_ss.copy(); x[1] = np.clip(h2r * pert, 0.05, 0.70)
            for _ in range(100):
                err = x[1] - h2r; u = u_ss.copy(); u[0] += np.clip(-0.6*err, -0.6, 0.6) + rng.uniform(-0.1, 0.1); u = np.clip(u, 0, 1)
                z = psi(x); xn = rk4(x, u); zn = psi(xn)
                z_k.append(z); u_k.append(u); z_k1.append(zn); x = xn
    for (h2f, h2t) in [(0.10, 0.20), (0.20, 0.15), (0.15, 0.10)]:
        xf, uf = compute_ss(h2f); xt, _ = compute_ss(h2t)
        if xf is None or xt is None: continue
        x = xf.copy()
        for _ in range(200):
            err = x[1] - h2t; u = uf.copy(); u[0] += np.clip(-0.5*err, -0.5, 0.5) + rng.uniform(-0.08, 0.08); u = np.clip(u, 0, 1)
            z = psi(x); xn = rk4(x, u); zn = psi(xn)
            z_k.append(z); u_k.append(u); z_k1.append(zn); x = xn
    Z = np.array(z_k); U = np.array(u_k); Zn = np.array(z_k1); nz, nu = Z.shape[1], U.shape[1]
    Phi = np.hstack([Z, U]); Theta = np.linalg.solve(Phi.T @ Phi + 1e-4 * np.eye(nz + nu), Phi.T @ Zn)
    A_raw = Theta[:nz, :].T; B_d = Theta[nz:, :].T
    vals, vecs = np.linalg.eig(A_raw); vals_s = np.where(np.abs(vals) >= 1.0, vals / np.abs(vals) * (1.0 - 1e-4), vals)
    A_d = np.real(vecs @ np.diag(vals_s) @ np.linalg.inv(vecs))
    if verbose: print(f"  Entrenadas {len(Z)} muestras | rho(A) = {np.max(np.abs(np.linalg.eigvals(A_d))):.5f}")
    return A_d, B_d, nz, nu

def design_observer(A_d, nz):
    C_obs = np.zeros((4, nz)); C_obs[0, 0] = 1.0; C_obs[1, 2] = 1.0; C_obs[2, 6] = 1.0; C_obs[3, 7] = 1.0
    for alpha in [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        try:
            X = la.solve_discrete_are(A_d.T, C_obs.T, alpha * np.eye(nz), np.eye(4))
            L_c = np.linalg.solve(C_obs @ X @ C_obs.T + np.eye(4), C_obs @ X @ A_d.T).T
            if np.abs(np.linalg.eigvals(A_d - L_c @ C_obs)).max() < 0.999: return L_c, C_obs
        except: pass
    return np.zeros((nz, 4)), C_obs

class EKF:
    C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    def __init__(self, x0, P0, Q, R, p=P_NOM):
        self.x = np.array(x0, dtype=float); self.P = P0.copy(); self.Q = Q.copy(); self.R = R.copy(); self.p = p
    def _jacobian(self, x, u, eps=1e-6):
        n = len(x); J = np.zeros((n, n)); f0 = nl_ode(x, u, self.p)
        for i in range(n):
            xp = x.copy(); xp[i] += eps; J[:, i] = (nl_ode(xp, u, self.p) - f0) / eps
        return np.eye(n) + Ts * J
    def predict(self, u):
        Fk = self._jacobian(self.x, u); self.x = rk4(self.x, u, self.p, Ts)
        self.x = np.clip(self.x, 0.0, 0.75); self.P = Fk @ self.P @ Fk.T + self.Q
    def update(self, y_meas):
        S = self.C @ self.P @ self.C.T + self.R; K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array(y_meas, float) - self.C @ self.x)
        self.x = np.clip(self.x, 0.0, 0.75); self.P = (np.eye(3) - K @ self.C) @ self.P; self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

# ============================================================================
# UTILIDADES Y LOGS
# ============================================================================
class _Tee2:
    def __init__(self, *fs): self.fs = fs
    def write(self, s):
        for f in self.fs: f.write(s)
    def flush(self):
        for f in self.fs: f.flush()

def _hdr(t): print("\n" + "=" * 104); print(" " + t); print("=" * 104)

def _fm(x):
    x = np.asarray(x, float); f = x[np.isfinite(x)]
    if len(f) == 0: return "n/d"
    return f"{f.mean():.3f}±{f.std(ddof=1):.3f}" if len(f) > 1 else f"{f.mean():.3f}"

def mask_paper(T):
    m = np.ones(len(T), dtype=bool)
    for tc in (0.0, 264.0, 528.0): m &= ~((T >= tc) & (T < tc + SETTLING_S))
    return m

def outlier_filter(x, max_delta=MAX_DELTA_H):
    y = np.array(x, dtype=float)
    for k in range(1, len(y)):
        if abs(y[k] - y[k-1]) > max_delta: y[k] = y[k-1]
    return y

def run_num(p):
    m = re.search(r'run(\d+)', os.path.basename(p)); return int(m.group(1)) if m else -1

def find_logs(csv_dir, prefix):
    return sorted(glob.glob(os.path.join(csv_dir, f'{prefix}_run*.csv')), key=run_num)

def load_log(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    est = next(c for c in ('h2_est_m', 'h2_ekf_m') if c in df.columns)
    return dict(T=df['t_s'].values, h1=df['h1_meas_m'].values, h3=df['h3_meas_m'].values,
                h2r=df['h2_real_m'].values, ref=df['ref_m'].values,
                u=np.column_stack([df['u1'].values, df['u3'].values]), h2_log=df[est].values)

def _pre(logs): return [(d, outlier_filter(d['h1']), outlier_filter(d['h3'])) for d in logs]

def _obs_stats(d, h2e):
    e = ((h2e - d['h2r']) * 100)[mask_paper(d['T'])]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e)), float(np.std(e, ddof=1))

def replay_koopman(d, A, B, L, C, h1=None, h3=None):
    h1 = d['h1'] if h1 is None else h1; h3 = d['h3'] if h3 is None else h3
    n = len(d['T']); z = psi(np.array([h1[0], d['ref'][0], h3[0]])); uprev = np.zeros(2)
    X = np.zeros((n, 3)); Z = np.zeros((n, A.shape[0])); X[0] = [h1[0], d['ref'][0], h3[0]]; Z[0] = z
    for k in range(n - 1):
        zp = A @ z + B @ uprev; zp[:3] = np.maximum(zp[:3], 0); zp[6:] = np.maximum(zp[6:], 0)
        y = np.array([np.sqrt(max(h1[k+1], EPS_PSI)), np.sqrt(max(h3[k+1], EPS_PSI)), h1[k+1], h3[k+1]])
        z = zp + L @ (y - C @ zp); z[:3] = np.maximum(z[:3], 0); z[6:] = np.maximum(z[6:], 0)
        X[k+1] = np.clip(z[:3]**2, 0, 0.75); Z[k+1] = z; uprev = d['u'][k].copy()
    return X, Z

def _ekf_rep(d, h1, h3, Q, R):
    n = len(d['T']); ekf = EKF(x0=np.array([h1[0], d['ref'][0], h3[0]]), P0=EKF_P0, Q=Q, R=R)
    uprev = np.zeros(2); X = np.zeros((n, 3)); X[0] = ekf.x
    for k in range(n - 1):
        ekf.predict(uprev); X[k + 1] = ekf.update(np.array([h1[k + 1], h3[k + 1]])); uprev = d['u'][k].copy()
    return X

def _design_L2(A, C, nz, aM, aU):
    q = np.full(nz, aU, float); q[[0, 2, 6, 7]] = aM          # aM: estados ligados a lo medido; aU: a lo no medido
    X = la.solve_discrete_are(A.T, C.T, np.diag(q), np.eye(4))
    return np.linalg.solve(C @ X @ C.T + np.eye(4), C @ X @ A.T).T

# ============================================================================
# AFINACIÓN Y VALIDACIÓN (S1 a S4)
# ============================================================================
def tuning_cv(K, E, A, B, C, nz):
    _hdr("T1. AFINACIÓN JUSTA: ambos observadores con 2-3 parámetros, afinados con VALIDACIÓN CRUZADA (RMSE obs, ventana única)")
    logs = K + E; pre = _pre(logs); n = len(pre)
    ekf_cfgs = [(s13, sh2, r) for s13 in (0.1, 0.5, 2.0) for sh2 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 6.0) for r in (0.5, 1.5, 4.0)]
    kp_cfgs = [(aM, aU) for aM in (1e-3, 1e-2, 0.1, 1.0, 10.0) for aU in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)]
    Me = np.full((len(ekf_cfgs), n), np.nan); Be = Me.copy(); Se = Me.copy()
    Mk = np.full((len(kp_cfgs), n), np.nan); Bk = Mk.copy(); Sk = Mk.copy()

    print(f"  Koopman: {len(kp_cfgs)} configuraciones (alpha_med, alpha_nomed) | EKF: {len(ekf_cfgs)} configuraciones (sigma_h1h3, sigma_h2, sigma_R) [mm]")
    for i, (aM, aU) in enumerate(kp_cfgs):
        try: L = _design_L2(A, C, nz, aM, aU)
        except Exception: continue
        if np.max(np.abs(np.linalg.eigvals(A - L @ C))) >= 1: continue
        for j, (d, h1, h3) in enumerate(pre):
            Mk[i, j], Bk[i, j], Sk[i, j] = _obs_stats(d, replay_koopman(d, A, B, L, C, h1, h3)[0][:, 1])
    print("  Koopman listo.", flush=True)
    
    for i, (s13, sh2, rr) in enumerate(ekf_cfgs):
        Q = np.diag([(s13 * 1e-3) ** 2, (sh2 * 1e-3) ** 2, (s13 * 1e-3) ** 2]); R = np.diag([(rr * 1e-3) ** 2] * 2)
        for j, (d, h1, h3) in enumerate(pre):
            Me[i, j], Be[i, j], Se[i, j] = _obs_stats(d, _ekf_rep(d, h1, h3, Q, R)[:, 1])
        if (i + 1) % 12 == 0: print(f"  EKF {i+1}/{len(ekf_cfgs)}", flush=True)

    ik = kp_cfgs.index((1.0, 1.0)); ie = ekf_cfgs.index((0.5, 2.0, 1.5))
    print(f"\n  CONFIGURACIÓN DE LOS EXPERIMENTOS (sobre las 10 corridas): Koopman alpha=1  -> {_fm(Mk[ik])} cm | EKF 0.5/2.0/1.5 mm -> {_fm(Me[ie])} cm")
    print("  (deben dar ~1.151 y ~1.387, como en S2)")

    def best(M): j = int(np.nanargmin(np.nanmean(M, axis=1))); return j
    jk, je = best(Mk), best(Me)
    print(f"\n  MEJOR GLOBAL (optimista: afinado y evaluado en los mismos datos):")
    print(f"    Koopman {kp_cfgs[jk]} -> RMSE {_fm(Mk[jk])} | sesgo {_fm(Bk[jk])} | sigma {_fm(Sk[jk])}")
    print(f"    EKF     {ekf_cfgs[je]} -> RMSE {_fm(Me[je])} | sesgo {_fm(Be[je])} | sigma {_fm(Se[je])}")
    edge = []
    if ekf_cfgs[je][1] in (0.02, 6.0): edge.append("sigma_h2 del EKF en el borde de la rejilla")
    if kp_cfgs[jk][1] in (1e-4, 10.0) or kp_cfgs[jk][0] in (1e-3, 10.0): edge.append("alpha de Koopman en el borde de la rejilla")
    if edge: print("    AVISO:", "; ".join(edge), "-> ampliar la rejilla.")

    def loo(M):
        v, ch = [], []
        for i in range(M.shape[1]):
            j = int(np.nanargmin(np.nanmean(np.delete(M, i, axis=1), axis=1))); v.append(M[j, i]); ch.append(j)
        return np.array(v), ch

    def fold2(M):
        tr_a, te_a = list(range(0, 5)), list(range(5, 10)); out = np.zeros(10); ch = []
        j = int(np.nanargmin(np.nanmean(M[:, tr_a], axis=1))); out[te_a] = M[j, te_a]; ch.append(j)
        j = int(np.nanargmin(np.nanmean(M[:, te_a], axis=1))); out[tr_a] = M[j, tr_a]; ch.append(j)
        return out, ch

    print("\n  RESULTADO HONESTO (RMSE obs [cm] en la corrida NO usada para afinar):")
    res = {}
    for name, fn in (("Leave-one-run-out (afina con 9, evalúa 1)", loo), ("2 bloques (afina con 5 corridas Koopman -> evalúa EKF, y viceversa)", fold2)):
        vk, chk = fn(Mk); ve, che = fn(Me); diff = ve - vk
        try: wp = stats.wilcoxon(ve, vk).pvalue
        except Exception: wp = float('nan')
        tp = stats.ttest_rel(ve, vk).pvalue
        print(f"   {name}")
        print(f"     Koopman afinado {_fm(vk)} | EKF afinado {_fm(ve)} | mejora de Koopman {(ve.mean()-vk.mean())/ve.mean()*100:6.1f}% | "
              f"Koopman mejor en {int((diff > 0).sum())}/10 | t pareado p={tp:.3g} | Wilcoxon p={wp:.3g}")
        print(f"     configuraciones elegidas Koopman: {sorted(set(kp_cfgs[c] for c in chk))} | EKF: {sorted(set(ekf_cfgs[c] for c in che))}")
        res[name] = (vk, ve)
    print("\n  LECTURA: si 'mejora de Koopman' ~0 o negativa con ambos protocolos, la ventaja de la Tabla 4 era de afinación del EKF,")
    print("  no del observador. Entonces el artículo debe decir 'comparable cuando ambos están afinados', y el mejor sustento")
    print("  es repetir experimentos de lazo cerrado con AMBOS afinados (ver recomendación).")
    return Mk, Me, kp_cfgs, ekf_cfgs

def _replay_cons(d, A, B, L, C, h1, h3, relift=False):
    n = len(d['T']); z = psi(np.array([h1[0], d['ref'][0], h3[0]])); uprev = np.zeros(2)
    X = np.zeros((n, 3)); Z = np.zeros((n, A.shape[0])); X[0] = [h1[0], d['ref'][0], h3[0]]; Z[0] = z; nclip = 0
    for k in range(n - 1):
        zp = A @ z + B @ uprev
        nclip += int(np.sum(zp[:3] < 0) + np.sum(zp[6:] < 0)); zp[:3] = np.maximum(zp[:3], 0); zp[6:] = np.maximum(zp[6:], 0)
        y = np.array([np.sqrt(max(h1[k + 1], EPS_PSI)), np.sqrt(max(h3[k + 1], EPS_PSI)), h1[k + 1], h3[k + 1]])
        z = zp + L @ (y - C @ zp)
        nclip += int(np.sum(z[:3] < 0) + np.sum(z[6:] < 0)); z[:3] = np.maximum(z[:3], 0); z[6:] = np.maximum(z[6:], 0)
        if relift: z = psi(np.clip(z[:3] ** 2, 0, 0.75))
        X[k + 1] = np.clip(z[:3] ** 2, 0, 0.75); Z[k + 1] = z; uprev = d['u'][k].copy()
    return X, Z, nclip

def consistencia(K, E, A, B, L, C):
    _hdr("T2. CONSISTENCIA ALGEBRAICA DE LOS OBSERVABLES ELEVADOS Y NO-NEGATIVIDAD (R2 'fairness', R3:7)")
    logs = K + E; rr = {'h1h2': [], 'h2h3': [], 'h1h3': []}; rel = []; clips = []; o_a, o_b = [], []
    for d in logs:
        h1 = outlier_filter(d['h1']); h3 = outlier_filter(d['h3'])
        X, Z, nc = _replay_cons(d, A, B, L, C, h1, h3, relift=False)
        r4 = Z[:, 3] - Z[:, 0] ** 2 * Z[:, 1] ** 2; r5 = Z[:, 4] - Z[:, 1] ** 2 * Z[:, 2] ** 2; r6 = Z[:, 5] - Z[:, 0] ** 2 * Z[:, 2] ** 2
        for k, r in zip(rr, (r4, r5, r6)): rr[k].append(float(np.sqrt(np.mean(r ** 2))))
        rel.append(float(np.sqrt(np.mean(r4 ** 2)) / np.sqrt(np.mean(Z[:, 3] ** 2)) * 100)); clips.append(nc)
        o_a.append(_obs_stats(d, X[:, 1])[0]); o_b.append(_obs_stats(d, _replay_cons(d, A, B, L, C, h1, h3, relift=True)[0][:, 1])[0])
    print(f"  Residuo RMS de las restricciones z4=z1^2 z2^2, z5=z2^2 z3^2, z6=z1^2 z3^2 [m^2]:")
    for k, v in rr.items(): print(f"    {k}: {np.mean(v):.2e} (máx entre corridas {np.max(v):.2e})")
    print(f"  Residuo de z4 relativo al RMS de z4: {np.mean(rel):.2f} %  (si es chico, el vector elevado es casi consistente)")
    print(f"  Recortes a cero aplicados (raíces/linealizaciones negativas) por corrida de 400 pasos: media {np.mean(clips):.1f}, máx {np.max(clips)}")
    print(f"  RMSE obs: observador tal como se usó {_fm(o_a)} | con re-elevación z=psi(h_est) en cada paso (restituye consistencia) {_fm(o_b)}")
    print("  Texto sugerido: el estado elevado se inicializa como psi(x0) (consistente); luego evoluciona linealmente y se corrige con las")
    print("  4 componentes medidas; sus restricciones no se imponen, solo z_i>=0 (recorte). Se reporta el residuo y el efecto de re-elevar.")

def _trans2(d):
    T = d['T']; gt = medfilt(d['h2r'], 5); out = []
    for i, (t0, t1, r) in enumerate(SEGS):
        idx = np.where((T >= t0) & (T < t1))[0]; seg = gt[idx]; tt = T[idx] - t0
        final = float(np.mean(d['h2r'][(T >= t1 - 60) & (T < t1)]))
        init = float(d['h2r'][0]) if i == 0 else float(np.mean(d['h2r'][(T >= t0 - 30) & (T < t0)]))
        step = final - init; s = np.sign(step); prog = (seg - init) / step
        t90 = float(tt[np.argmax(prog >= 0.9)]) if np.any(prog >= 0.9) else np.nan
        row = [t90]
        for band in (0.5, 1.0):
            bad = np.where(np.abs(seg - final) * 100 > band)[0]
            row.append(0.0 if len(bad) == 0 else (float(tt[bad[-1] + 1]) if bad[-1] + 1 < len(tt) else np.nan))
        os_cm = max(0.0, float(np.max(s * (seg - final)))) * 100
        row += [os_cm, os_cm / (abs(step) * 100) * 100, (final - r) * 100]
        out.append(row)
    return np.array(out)

def transitorios(K, E):
    _hdr("T3. TRANSITORIOS REDEFINIDOS (verdad terreno, mediana de 5 muestras) — para R2 'Evaluation windows'")
    print("  t90: tiempo a 90 % del escalón | Ts(±0.5 cm) y Ts(±1.0 cm): asentamiento en banda ABSOLUTA alrededor del valor FINAL")
    print("  (no del 5 % de la referencia: con escalones de 5-10 cm esa banda es de 2.5-5 mm, menor que el ruido/resolución del sensor)")
    print("  sobreimpulso = exceso sobre el valor final en el sentido del escalón (cm y % del escalón) | offset = final GT - referencia.")
    print("  n/5 = corridas que llegaron a asentarse. El sobreimpulso del seg. 3 del manuscrito (5-6 cm) era el nivel inicial, no sobreimpulso.")
    allr = {}
    for name, runs in (('Koopman', K), ('EKF', E)):
        tr = np.array([_trans2(d) for d in runs]); allr[name] = tr
        for s, (_, _, r) in enumerate(SEGS):
            c = tr[:, s, :]; ok5 = int(np.isfinite(c[:, 1]).sum()); ok10 = int(np.isfinite(c[:, 2]).sum())
            print(f"  {name:<8}Seg{s+1} ({r:.2f} m): t90 {_fm(c[:,0])} s | Ts0.5 {_fm(c[:,1])} s ({ok5}/5) | Ts1.0 {_fm(c[:,2])} s ({ok10}/5) | "
                  f"sobreimp {_fm(c[:,3])} cm ({_fm(c[:,4])} %) | offset {_fm(c[:,5])} cm")
    for s in range(3):
        p = stats.ttest_ind(allr['Koopman'][:, s, 0], allr['EKF'][:, s, 0], equal_var=False).pvalue
        print(f"  t90 Seg{s+1}: Welch p = {p:.4f}")

def incertidumbre(K, E):
    _hdr("T4. INCERTIDUMBRE: IC 95 % por corrida y ruido del sensor de referencia (R2 'Experimental confidence', R1:5)")
    ko, eo, kp, ep = [], [], [], []
    for d in K:
        m = mask_paper(d['T']); ko.append(_obs_stats(d, d['h2_log'])[0]); kp.append(float(np.sqrt(np.mean((((d['h2r'] - d['ref']) * 100)[m]) ** 2))))
    for d in E:
        m = mask_paper(d['T']); eo.append(_obs_stats(d, d['h2_log'])[0]); ep.append(float(np.sqrt(np.mean((((d['h2r'] - d['ref']) * 100)[m]) ** 2))))
    rng = np.random.default_rng(0)

    def ci(a, b, lab):
        a, b = np.array(a), np.array(b); d = b.mean() - a.mean()
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); t = stats.t.ppf(0.975, df)
        bs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(10000)]
        print(f"  {lab}: Koopman {a.mean():.3f} [{a.mean()-stats.t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)):.3f}, {a.mean()+stats.t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)):.3f}] | "
              f"EKF {b.mean():.3f} [{b.mean()-stats.t.ppf(.975,len(b)-1)*b.std(ddof=1)/np.sqrt(len(b)):.3f}, {b.mean()+stats.t.ppf(.975,len(b)-1)*b.std(ddof=1)/np.sqrt(len(b)):.3f}] | "
              f"diferencia EKF-Koopman {d:.3f} cm, IC95 Welch [{d-t*se:.3f}, {d+t*se:.3f}], bootstrap [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
    ci(ko, eo, "RMSE obs (como se corrió)  "); ci(kp, ep, "RMSE físico GT-ref           ")
    sg = []
    for d in K + E:
        for (t0, t1, r) in SEGS:
            m = (d['T'] >= t1 - 90) & (d['T'] < t1); y = d['h2r'][m] * 100; x = d['T'][m]
            sg.append(float(np.std(y - np.polyval(np.polyfit(x, y, 1), x), ddof=1)))
    print(f"  Ruido del sensor de referencia estimado EN LOS DATOS (desv. típica en los últimos 90 s de cada tramo, sin tendencia): "
          f"mediana {np.median(sg):.2f} cm, rango [{np.min(sg):.2f}, {np.max(sg):.2f}] cm  (incluye fluctuación real del nivel)")
    print("  Nota: cualquier offset de calibración del sensor de referencia afecta IGUAL a ambos observadores; por eso la DIFERENCIA entre")
    print("  métodos (misma verdad terreno, mismas corridas) es robusta a ese offset, aunque el RMSE absoluto no.")

def main_tuning():
    os.makedirs(_T_OUT, exist_ok=True)
    f = open(os.path.join(_T_OUT, 'salida_tuning.txt'), 'w', encoding='utf-8')
    old = sys.stdout
    sys.stdout = _Tee2(old, f)
    try:
        csv_dir = CSV_DIR_DEFAULT
        kf = find_logs(csv_dir, 'koopman'); ef = find_logs(csv_dir, 'ekf')
        if not kf or not ef: raise SystemExit(f"No encontré logs en {csv_dir}")
        K = [load_log(p) for p in kf]; E = [load_log(p) for p in ef]
        
        print(f"Cargados {len(K)} logs Koopman y {len(E)} EKF.")
        A, B, nz, nu = train_edmd(verbose=False); L, C = design_observer(A, nz)
        
        tuning_cv(K, E, A, B, C, nz)
        consistencia(K, E, A, B, L, C)
        transitorios(K, E)
        incertidumbre(K, E)
        
        print("\nListo. Pásame salida_tuning/salida_tuning.txt completo.")
    finally:
        sys.stdout = old; f.close()

main_tuning()