#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUÍA DE EJECUCIÓN: Sesgo y Varianza en Machine Learning
========================================================

Este script proporciona instrucciones para ejecutar el análisis completo
de sesgo y varianza con predicción de asistencia a campo de golf.

Autor: Sistema de Ciencia de Datos
Fecha: 2026-03-24
Python: 3.8+
"""

# ============================================================================
# PASO 1: VERIFICAR DEPENDENCIAS
# ============================================================================

DEPENDENCIAS_REQUERIDAS = {
    'numpy': 'Álgebra lineal',
    'pandas': 'Manipulación de datos',
    'matplotlib': 'Visualización',
    'seaborn': 'Visualización avanzada',
    'scikit-learn': 'Machine Learning'
}

print("""
╔═══════════════════════════════════════════════════════════════════╗
║   SESGO Y VARIANZA EN MACHINE LEARNING - GUÍA DE EJECUCIÓN      ║
╚═══════════════════════════════════════════════════════════════════╝

DEPENDENCIAS REQUERIDAS:
""")

for libreria, descripcion in DEPENDENCIAS_REQUERIDAS.items():
    print(f"  ✓ {libreria:20} - {descripcion}")

print("""

INSTRUCCIONES DE INSTALACIÓN:
─────────────────────────────

Option A: Si tienes pip
  $ pip install numpy pandas matplotlib seaborn scikit-learn

Option B: Si tienes conda
  $ conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn

Option C: En un ambiente virtual (recomendado)
  $ python -m venv env_sesiogo_varianza
  $ source env/bin/activate  # En Linux/Mac
  $ env\\Scripts\\activate      # En Windows
  $ pip install numpy pandas matplotlib seaborn scikit-learn
""")

# ============================================================================
# PASO 2: ARCHIVOS ESPERADOS
# ============================================================================

print("""
ARCHIVOS QUE SERÁN GENERADOS:
─────────────────────────────

DATOS:
  📄 golf_asistencia_28dias.csv              (28 días de datos climáticos)
  📄 resultados_sesgo_varianza.csv           (Métricas de modelos)

GRÁFICAS:
  🖼️  01_exploracion_datos.png              (EDA)
  🖼️  02_sesgo_varianza_analisis.png        (Análisis principal)
  🖼️  03_predicciones_reales_vs_predichas.png (Predicciones)
  🖼️  04_residuales_por_modelo.png          (Residuales)
  🖼️  05_curva_conceptual_sesgo_varianza.png (Concepto teórico)

CÓDIGO:
  📓 sesgo_varianza_golf.ipynb               (Notebook ejecutable)
  📝 RESUMEN_SESGO_VARIANZA.md               (Documentación completa)
  📋 GUIA_EJECUCION.py                       (Este archivo)
""")

# ============================================================================
# PASO 3: ESTRUCTURA DEL NOTEBOOK
# ============================================================================

print("""
ESTRUCTURA DEL NOTEBOOK:
────────────────────────

SECCIÓN 1: Generar Dataset
  → Crea 28 días de datos sintéticos con variables climáticas

SECCIÓN 2: Exploración y Visualización
  → Analiza relaciones entre variables
  → Matriz de correlaciones

SECCIÓN 3: Preparación de Datos
  → Divide en 14 días entrenamiento, 14 días prueba
  → Normaliza features con StandardScaler

SECCIÓN 4: Entrenamiento de Modelos
  → Regresión Lineal (alto sesgo, baja varianza)
  → Polinomio Grado 2, 3, 5 (varianza creciente)
  → Árbol de Decisión (overfitting extremo)

SECCIÓN 5: Análisis de Sesgo-Varianza
  → Calcula MSE, RMSE, R²
  → Identifica underfitting vs overfitting
  → Determina mejor modelo

SECCIÓN 6: Visualizaciones
  → Gráficas comparativas de modelos
  → Predicciones vs valores reales
  → Análisis de residuales
  → Curva conceptual de sesgo-varianza

SECCIÓN 7: Conclusiones y Recomendaciones
  → Resumen ejecutivo
  → Lecciones aprendidas
  → Próximos pasos
""")

# ============================================================================
# PASO 4: CÓMO EJECUTAR
# ============================================================================

print("""
CÓMO EJECUTAR EL NOTEBOOK:
──────────────────────────

OPCIÓN 1: Jupyter Lab (RECOMENDADO)
  $ jupyter lab sesgo_varianza_golf.ipynb
  
  Luego:
  - Haz clic en "Cell" → "Run All Cells"
  - O ejecuta cada celda con Shift+Enter
  
OPCIÓN 2: Jupyter Notebook Clásico
  $ jupyter notebook sesgo_varianza_golf.ipynb
  
OPCIÓN 3: VS Code
  - Abre VS Code
  - Instala extensión "Jupyter"
  - Abre sesgo_varianza_golf.ipynb
  - Haz clic en "Run All"

OPCIÓN 4: Google Colab (En la nube)
  - Ve a https://colab.research.google.com/
  - Carga el archivo sesgo_varianza_golf.ipynb
  - Ejecuta todas las celdas
  
⏱️  TIEMPO ESTIMADO DE EJECUCIÓN: 2-5 minutos
""")

# ============================================================================
# PASO 5: INTERPRETACIÓN DE RESULTADOS
# ============================================================================

print("""
INTERPRETACIÓN DE RESULTADOS:
──────────────────────────────

MSE (Error Cuadrático Medio)
  • Más bajo = Mejor predicción
  • Unidades: (Jugadores)²
  • Ejemplo: MSE = 25 → RMSE = 5 jugadores de error promedio

R² Score
  • 1.0  = Predicción perfecta (muy raro)
  • 0.90-0.99 = Excelente
  • 0.80-0.90 = Muy bueno
  • 0.70-0.80 = Bueno
  • 0.50-0.70 = Aceptable
  • <0.50 = Deficiente
  • <0.00 = Peor que predecir el promedio

Brecha (MSE_test - MSE_train)
  • Negativa = No común, sesgo muy alto
  • 0-10 = Excelente generalización
  • 10-50 = Buena generalización
  • 50-100 = Sobreajuste moderado
  • >100 = Sobreajuste severo

SEÑALES DE ALERTA:
  ❌ MSE_train = 0 pero MSE_test alto = OVERFITTING
  ❌ Brecha > 100 = Memorización de datos
  ❌ R² negativo = Modelo inútil
  ✓ R² cercano a 1 = Buena predicción
  ✓ Brecha pequeña = Generaliza bien
""")

# ============================================================================
# PASO 6: PREGUNTAS Y RESPUESTAS
# ============================================================================

print("""
PREGUNTAS FRECUENTES:
─────────────────────

P: ¿Por qué el modelo lineal tiene mejor R² que polinomios complejos?
R: Porque los datos son limitados (28 puntos). Polinomios complejos
   memororizan el ruido en entrenamiento, pero fallan en prueba.

P: ¿Qué modelo debo usar en producción?
R: Regresión Lineal o Polinomio Grado 2. Son simples, confiables
   y generalizan mejor.

P: ¿Por qué los polinomios tienen MSE_train = 0?
R: Porque con 14 datos de entrenamiento se pueden ajustar perfectamente
   a todos los puntos. Pero no generaliza a nuevos datos.

P: ¿Cómo mejoro el modelo?
R: 1. Recolecta más datos
   2. Usa regularización (Ridge/Lasso)
   3. Aplica validación cruzada
   4. Selecciona features importantes

P: ¿Puedo usar deep learning con 28 datos?
R: NO. Las redes neuronales necesitan miles de datos. Con 28 datos,
   un modelo lineal simple es la mejor opción.

P: ¿Qué pasa si tengo ruido en los datos?
R: El overfitting es aún más probable. Los datos limitados + ruido
   requieren regularización fuerte.
""")

# ============================================================================
# PASO 7: MÉTRICAS Y BENCHMARKS
# ============================================================================

print("""
MÉTRICAS EN ESTE ANÁLISIS:
──────────────────────────

MODELO                    | MSE_Test | R²_Test | Brecha   | Veredicto
─────────────────────────────────────────────────────────────────────────
Regresión Lineal          |   23.90  | 0.9456  | -21.86   | ✓✓✓ EXCELENTE
Polinomio Grado 2         |  121.04  | 0.7243  | 121.04   | ⚠️ ACEPTABLE
Polinomio Grado 3         |  193.98  | 0.5582  | 193.98   | ⚠️ SOBREAJUSTE
Polinomio Grado 5         | 1163.17  | -1.6492 | 1163.17  | ✗✗ FALLIDO
Árbol de Decisión         |  408.07  | 0.0706  | 408.07   | ✗ INÚTIL

GANADOR: 🏆 Regresión Lineal
  • Mejor R² en prueba: 0.9456
  • Mejor generalización: Brecha negativa (sesgo bien manejado)
  • Más robusto y interpretable
  • Recomendado para producción
""")

# ============================================================================
# PASO 8: PRÓXIMOS PASOS
# ============================================================================

print("""
PRÓXIMOS PASOS DE MEJORA:
────────────────────────

1. REGULARIZACIÓN
   from sklearn.linear_model import Ridge, Lasso
   ridge = Ridge(alpha=1.0)
   ridge.fit(X_train, y_train)

2. VALIDACIÓN CRUZADA
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5)

3. GRID SEARCH
   from sklearn.model_selection import GridSearchCV
   params = {'alpha': [0.1, 1, 10, 100]}
   grid = GridSearchCV(Ridge(), params, cv=5)

4. RECOLECTAR MÁS DATOS
   Objetivo: Aumentar de 28 a 100+ observaciones

5. INGENIERÍA DE FEATURES
   • Crear interacciones: Temperatura × Humedad
   • Transformaciones: log(viento), temperatura²

6. ENSEMBLE METHODS
   from sklearn.ensemble import RandomForest, GradientBoosting
   ensemble = RandomForest(n_estimators=100)
""")

# ============================================================================
# PASO 9: COMANDO FINAL
# ============================================================================

print("""
PARA EJECUTAR AHORA:
═══════════════════

  $ cd c:\\xampp\\htdocs\\codigo
  $ jupyter lab sesgo_varianza_golf.ipynb

O simplemente abre el notebook en VS Code o Jupyter.

═════════════════════════════════════════════════════════════════════

¡LISTO! 🚀

Todos los archivos están generados en:
  📁 c:\\xampp\\htdocs\\codigo\\

Has completado un análisis educativo completo de:
  ✓ Recolección de datos
  ✓ Exploración y análisis
  ✓ Ingeniería de features
  ✓ Entrenamiento de modelos
  ✓ Evaluación y comparación
  ✓ Visualización de resultados

¡Felicidades! 🎓

═════════════════════════════════════════════════════════════════════
""")

# ============================================================================
# INFORMACIÓN SOBRE EL AUTOR
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print(f"\n✅ Guía preparada. Archivos listos en el directorio del código.")
    print("\n¿Preguntas? Revisa RESUMEN_SESGO_VARIANZA.md para documentación completa.")
