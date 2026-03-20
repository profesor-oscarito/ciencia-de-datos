from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Crear documento PDF
pdf_path = r'c:\Users\ocog1\OneDrive\Duoc UC\ciencia de datos con ython\probando codigo\actividad11.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter)
story = []
styles = getSampleStyleSheet()

# Agregar contenido del notebook al PDF
story.append(Paragraph('Actividad 1.1. Introduccion a Estructuras de Datos', styles['Heading1']))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph('Temas a cubrir', styles['Heading2']))
story.append(Paragraph('- Listas (list)<br/>- Diccionarios (dict)<br/>- Arrays (Numpy)<br/>- DataFrames (Pandas)', styles['Normal']))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph('<b>Importante:</b> Revisar los datos antes de ejecutar el modelo.', styles['Normal']))
story.append(Spacer(1, 0.1*inch))

story.append(Paragraph('<b>Tip:</b> Usa pandas para limpiar datos rapidamente.', styles['Normal']))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph('Codigo:', styles['Heading2']))
story.append(Paragraph('# Ejemplo basico de estructuras de datos<br/>for i in range(5):<br/>&nbsp;&nbsp;&nbsp;&nbsp;print(i)', styles['Normal']))

# Generar PDF
doc.build(story)
print(f'PDF creado exitosamente: {pdf_path}')
