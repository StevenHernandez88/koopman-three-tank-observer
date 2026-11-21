from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

out_path = r"c:\Users\PC\Downloads\articulo\Resumen_Book_for_writing_research_papers.docx"
doc = Document()

# Style
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# Title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('Resumen del libro: English for Writing Research Papers')
run.bold = True
run.font.size = Pt(16)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Guía práctica para escribir artículos académicos en español e inglés').italic = True

# Intro
p = doc.add_paragraph()
p.add_run('Este documento resume los puntos más importantes del libro de Adrian Wallwork para ayudar a investigadores no nativos a redactar artículos científicos con mayor claridad, precisión y coherencia.')

# Spanish section
sec = doc.add_heading('Resumen en español', level=1)

sub = doc.add_heading('1. Idea central del libro', level=2)
paragraphs = [
    'El libro enseña cómo escribir papers de manera clara y aceptable para revistas internacionales.',
    'Enfatiza que la calidad de un artículo no depende solo del contenido, sino también de la estructura, el orden de ideas, la concisión y la legibilidad.',
    'La recomendación principal es escribir con foco en el lector: el texto debe ser fácil de seguir, sin ambigüedades ni redundancias.'
]
for text in paragraphs:
    p = doc.add_paragraph(text)

sub = doc.add_heading('2. Puntos clave para escribir mejor', level=2)
bullets = [
    'Planificar antes de escribir: definir objetivo, audiencia, revista y mensajes clave.',
    'Usar una estructura lógica: idea general → detalle → conclusión.',
    'Organizar frases y párrafos con orden claro y progresivo.',
    'Evitar oraciones demasiado largas y frases con demasiadas comas o cláusulas relativas.',
    'Ser conciso: eliminar palabras innecesarias y reducir redundancias.',
    'Usar conectores y señaladores para guiar al lector.',
    'Priorizar la claridad sobre la elegancia; un texto simple, preciso y bien estructurado es mejor.'
]
for item in bullets:
    doc.add_paragraph(item, style='List Bullet')

sub = doc.add_heading('3. Qué incluir en cada sección del paper', level=2)

sections = [
    ('Título', ['Debe ser breve, claro y específico. Debe reflejar el tema principal y, si es posible, el aporte más importante del estudio.']),
    ('Resumen (Abstract)', ['Debe resumir: contexto, objetivo, método, resultados principales y conclusión. Debe ser preciso y no repetir información detallada del cuerpo del artículo.']),
    ('Introducción', ['Presentar el problema, el contexto académico y la brecha de conocimiento. Explicar por qué el estudio es relevante y terminar con el objetivo o hipótesis del trabajo.']),
    ('Método / Metodología', ['Explicar cómo se hizo el estudio con suficiente detalle para que otro investigador pueda reproducirlo. Incluir diseño, participantes o datos, instrumentos, procedimiento y análisis.']),
    ('Resultados', ['Reportar los hallazgos de manera objetiva. No interpretar demasiado; mostrar datos relevantes, tendencias y resultados principales. Usar tablas y figuras cuando ayuden a la comprensión.']),
    ('Discusión', ['Interpretar los resultados, compararlos con trabajos previos y explicar su significado. También se deben mencionar limitaciones, implicaciones y posibles explicaciones.']),
    ('Conclusiones', ['Responder de forma breve a la pregunta de investigación. Resumir la contribución principal y, si aplica, señalar líneas futuras.']),
    ('Referencias', ['Usar un estilo de citación consistente y completo. Verificar que cada cita aparezca en la lista y viceversa.']),
    ('Figuras y tablas', ['Cada una debe ser autocontenida, con título claro y una leyenda que permita entenderla sin leer el texto completo.']),
    ('Limitaciones y agradecimientos', ['Reconocer limitaciones de manera honesta y, si corresponde, agradecer a quienes apoyaron el trabajo.'])
]
for title, items in sections:
    doc.add_paragraph(title, style='List Bullet 2')
    for item in items:
        doc.add_paragraph(item, style='List Bullet')

sub = doc.add_heading('4. Recomendaciones de estilo', level=2)
bullets = [
    'Usar voz activa cuando sea posible.',
    'Preferir frases cortas y claras a frases largas y complejas.',
    'No repetir ideas en varias partes del texto.',
    'Mantener el foco en los hallazgos más importantes.',
    'Revisar el texto para corregir errores de gramática, conectores y precisión terminológica.'
]
for item in bullets:
    doc.add_paragraph(item, style='List Bullet')

# English section
sec = doc.add_heading('Summary in English', level=1)

sub = doc.add_heading('1. Main idea of the book', level=2)
paragraphs = [
    'The book provides practical guidance for non-native researchers who want to write research papers in clear and publishable English.',
    'Its main message is that a paper is not judged only by its scientific value, but also by its readability, organization, and precision.',
    'The reader should always be kept in mind: the text must be easy to follow, logically organized, and free of unnecessary repetition.'
]
for text in paragraphs:
    p = doc.add_paragraph(text)

sub = doc.add_heading('2. Key writing principles', level=2)
bullets = [
    'Plan before drafting: define the aim, audience, journal, and key message.',
    'Use a clear structure: general idea → supporting details → conclusion.',
    'Organize sentences and paragraphs in a logical sequence.',
    'Avoid overly long sentences and excessive use of commas or relative clauses.',
    'Be concise: remove redundant words and avoid filler expressions.',
    'Use signposting and logical connectors to guide the reader.',
    'Prioritize clarity over stylistic complexity.'
]
for item in bullets:
    doc.add_paragraph(item, style='List Bullet')

sub = doc.add_heading('3. What to include in each section', level=2)
sections_en = [
    ('Title', ['It should be brief, informative, and specific. It should reflect the main topic and the main contribution of the study.']),
    ('Abstract', ['It should summarize the background, objective, method, main results, and conclusion. It should be concise and avoid repeating the full paper.']),
    ('Introduction', ['Present the problem, the research context, and the knowledge gap. Explain why the study matters and end with the aim or hypothesis.']),
    ('Methods', ['Describe the study clearly enough for replication. Include design, participants or data, instruments, procedure, and analysis.']),
    ('Results', ['Report the findings objectively. Do not over-interpret them; highlight the main outcomes and support them with figures or tables when useful.']),
    ('Discussion', ['Interpret the findings, compare them with previous work, and explain their significance. Mention limitations, implications, and possible explanations.']),
    ('Conclusion', ['Briefly answer the research question. Summarize the main contribution and, if relevant, suggest future work.']),
    ('References', ['Use a consistent citation style and verify that all sources are complete and accurately cited.']),
    ('Figures and tables', ['Each figure or table should be self-contained and clearly labeled so it can be understood without reading the whole paper.']),
    ('Limitations and acknowledgements', ['State limitations honestly and acknowledge any support received.'])
]
for title, items in sections_en:
    doc.add_paragraph(title, style='List Bullet 2')
    for item in items:
        doc.add_paragraph(item, style='List Bullet')

sub = doc.add_heading('4. Style recommendations', level=2)
bullets = [
    'Use active voice when possible.',
    'Prefer clear short sentences over complex ones.',
    'Avoid repetition and unnecessary hedging.',
    'Keep the focus on the main findings.',
    'Revise grammar, connectors, and terminology carefully.'
]
for item in bullets:
    doc.add_paragraph(item, style='List Bullet')

# Final note
p = doc.add_paragraph()
p.add_run('Conclusión: el libro es una guía útil para transformar un manuscrito técnico en un texto académico claro, bien estructurado y más probables de ser aceptado por revistas internacionales.').bold = True

# Save
try:
    doc.save(out_path)
    print('DOCX created:', out_path)
except Exception as e:
    print('ERROR', e)
