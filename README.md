# Sesgo y Varianza en Machine Learning 🎯
## Análisis Completo: Predicción de Asistencia a Campo de Golf

---

## 📊 Resumen Ejecutivo

Este proyecto demuestra el **trade-off fundamental** entre sesgo y varianza usando un caso práctico: predecir asistencia a un campo de golf con solo **28 días de datos**.

### Hallazgos Clave

| Métrica | Mejor Modelo | Valor |
|---------|--------------|-------|
| 🏆 Mejor R² | Regresión Lineal | 0.9456 |
| 📉 Menor MSE | Regresión Lineal | 23.90 |
| ✅ Mejor Generalización | Regresión Lineal | Brecha: -21.86 |

**Conclusión**: Modelo simple > Modelo complejo con datos limitados

---

## 📁 Archivos Generados

### 📊 Datos
```
golf_asistencia_28dias.csv
├─ 28 observaciones (días)
├─ Variables: Temperatura, Humedad, Viento, Tipo_Clima
└─ Target: Número de Jugadores

resultados_sesgo_varianza.csv
├─ 5 filas (modelos)
└─ 10 columnas (métricas)
```

### 📈 Visualizaciones
```
01_exploracion_datos.png
├─ Temperatura vs Jugadores
├─ Tipo Clima vs Jugadores
├─ Humedad vs Jugadores
└─ Viento vs Jugadores

02_sesgo_varianza_analisis.png
├─ MSE Train vs Test
├─ Indicador de Varianza (Brecha)
├─ R² Score por Modelo
└─ Explicaciones Conceptuales

03_predicciones_reales_vs_predichas.png
├─ Scatter plots para cada modelo
└─ Línea de predicción perfecta como referencia

04_residuales_por_modelo.png
├─ Distribución de errores
└─ Análisis de patrones

05_curva_conceptual_sesgo_varianza.png
├─ Curva teórica de Sesgo-Varianza
├─ Punto óptimo marcado
└─ Regiones de underfitting/overfitting
```

### 📓 Código y Documentación
```
sesgo_varianza_golf.ipynb
├─ 23 celdas (11 código, 12 markdown)
└─ Ejecutable en Jupyter

RESUMEN_SESGO_VARIANZA.md
├─ 400+ líneas de documentación
├─ Explicaciones teóricas
└─ Guía de interpretación

GUIA_EJECUCION.py
├─ Instrucciones paso a paso
├─ Preguntas frecuentes
└─ Próximos pasos

README.md (este archivo)
├─ Referencia rápida
└─ Links a recursos
```

---

## 🎯 Modelos Probados

### 1️⃣ Regresión Lineal
- **Sesgo**: Alto - **Varianza**: Baja
- **MSE Prueba**: 23.90 - **R²**: 0.9456
- **Diagnóstico**: ✅ **EXCELENTE** - Mejor opción

### 2️⃣ Polinomio Grado 2
- **Sesgo**: Moderado - **Varianza**: Moderada
- **MSE Prueba**: 121.04 - **R²**: 0.7243
- **Diagnóstico**: ⚠️ **Aceptable** - Algunos overfitting

### 3️⃣ Polinomio Grado 3
- **Sesgo**: Bajo - **Varianza**: Moderada-Alta
- **MSE Prueba**: 193.98 - **R²**: 0.5582
- **Diagnóstico**: ⚠️ **Overfitting moderado**

### 4️⃣ Polinomio Grado 5
- **Sesgo**: Muy Bajo - **Varianza**: Muy Alta
- **MSE Prueba**: 1163.17 - **R²**: -1.6492
- **Diagnóstico**: ❌ **Fallo catastrófico**

### 5️⃣ Árbol de Decisión
- **Sesgo**: Muy Bajo - **Varianza**: Muy Alta
- **MSE Prueba**: 408.07 - **R²**: 0.0706
- **Diagnóstico**: ❌ **Inútil en prueba**

---

## 💡 Conceptos Fundamentales

### Sesgo (Bias)
**Definición**: Error por asumir excesivas simplificaciones.

```
Real:     ~~~~ (curva ondulante)
Modelo:   ─── (línea recta)
Resultado: ❌ Siempre hay error
```

**Síntoma**: MSE_train ALTO, MSE_test ALTO (similar)

### Varianza (Variance)
**Definición**: Sensibilidad del modelo a fluctuaciones en datos.

```
Real:     ~~~~ (curva suave)
Modelo:   ╱╲╱╲╱╲ (zigzag)
Resultado: ✓ Train perfecto, ✗ Test terrible
```

**Síntoma**: MSE_train BAJO, MSE_test ALTO (brecha grande)

### Trade-off
**Ecuación**: 
$$\text{Error} = \text{Sesgo}^2 + \text{Varianza} + \text{Ruido}$$

```
Complejidad ↑
    ├─ Baja  → Alto Sesgo, Baja Varianza (Underfitting)
    ├─ Media → Bajo Sesgo, Baja Varianza (Óptimo) ✅
    └─ Alta  → Bajo Sesgo, Alta Varianza (Overfitting)
```

---

## 🚀 Cómo Usar

### Opción 1: Jupyter Lab (Recomendado)
```bash
$ cd c:\xampp\htdocs\codigo
$ jupyter lab sesgo_varianza_golf.ipynb
```

### Opción 2: VS Code
- Abre VS Code
- Click derecho en el archivo → Open with...
- Selecciona Jupyter Notebook

### Opción 3: Google Colab
- Sube el archivo a Google Colab
- Ejecuta todas las celdas

### Opción 4: Línea de comando
```bash
$ python GUIA_EJECUCION.py  # Ver instrucciones
```

---

## 📊 Interpretación Rápida

| Métrica | Buena | Mala |
|---------|-------|------|
| **R²** | > 0.70 | < 0.50 |
| **Brecha** | < 20 | > 100 |
| **MSE_train** | ~30 | ~0 (sospechoso) |
| **RMSE** | Bajo | Alto |

---

## 🎓 Lecciones Clave

1. ✅ **Modelos simples generalizan mejor con datos limitados**
2. ✅ **La brecha train-test es un indicador crítico de overfitting**  
3. ✅ **Un modelo perfecto en train es una red flag**
4. ✅ **Validación cruzada es obligatoria con pocos datos**
5. ✅ **Regularización ayuda a reducir varianza**

---

## 🔧 Próximas Mejoras

Para productizar este modelo:

- [ ] Recolectar más datos (objetivo: 100+ observaciones)
- [ ] Aplicar regularización (Ridge/Lasso)
- [ ] Validación cruzada k-fold
- [ ] Ingeniería de features (interacciones)
- [ ] Tuning de hiperparámetros
- [ ] Monitoreo en producción

---

## 📚 Referencias

- **Libro**: "The Elements of Statistical Learning" (Hastie et al., 2009)
- **Libro**: "An Introduction to Statistical Learning" (James et al., 2013)
- **Artículo**: "A Few Useful Things to Know about Machine Learning" (Domingos, 2012)
- **Curso**: Andrew Ng's ML Course (Coursera)

---

## 👤 Información del Proyecto

- **Tema**: Sesgo y Varianza en Machine Learning
- **Caso**: Predicción de asistencia a campo de golf
- **Dataset**: 28 observaciones (28 días)
- **Modelos**: 5 (Lineal, Poly2, Poly3, Poly5, Decision Tree)
- **Gráficas**: 5 visualizaciones principales
- **Lenguaje**: Python 3.8+
- **Frameworks**: Scikit-learn, Pandas, Matplotlib

---

## ✅ Checklist de Ejecución

- [ ] Instalar dependencias: `pip install numpy pandas matplotlib seaborn scikit-learn`
- [ ] Abrir el notebook `sesgo_varianza_golf.ipynb`
- [ ] Ejecutar todas las celdas (Shift+Enter)
- [ ] Revisar las 5 gráficas generadas
- [ ] Leer el resumen en `RESUMEN_SESGO_VARIANZA.md`
- [ ] Estudiar las conclusiones
- [ ] Aplicar conceptos a tus propios problemas

---

## 💬 Preguntas Frecuentes

**P: ¿Por qué regresión lineal es mejor que polinomios complejos?**
R: Porque con solo 28 datos, los modelos complejos memoriran ruido. La regresión lineal generaliza mejor.

**P: ¿Qué puedo hacer si tengo overfitting?**
R: Usar regularización (Ridge/Lasso), obtener más datos, o seleccionar features importantes.

**P: ¿Este análisis aplica a otros datasets?**
R: Sí, los conceptos de sesgo-varianza son universales. El trade-off existe en todo ML.

**P: ¿Cómo sé cuándo mi modelo está overfitting?**
R: Mira la brecha train-test. Si está > 100, hay overfitting severo.

---

## 🎉 Conclusión

**Con datos limitados, simplicidad es poder.**

Este proyecto demuestra que un modelo lineal simple puede superar a modelos complejos cuando:
- Los datos son limitados
- El overfitting es un riesgo
- La interpretabilidad importa

**Principio de Occam**: No multipliques las entidades más de lo necesario. 

---

**¡Gracias por revisar este análisis!** 🚀

Para más detalles, abre `RESUMEN_SESGO_VARIANZA.md`

---

*Última actualización: 2026-03-24*
