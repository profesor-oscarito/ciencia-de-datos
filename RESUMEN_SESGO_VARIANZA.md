# 📊 Análisis Completo: Sesgo y Varianza en Machine Learning
## Predicción de Asistencia a Campo de Golf

---

## 📋 Contenido del Proyecto

Este proyecto de aprendizaje demostrativo incluye:

1. **Dataset Sintético**: 28 días de datos climáticos y asistencia de jugadores
2. **5 Modelos de Regresión**: Con diferentes niveles de complejidad
3. **Análisis Detallado**: Comparación de sesgo vs varianza
4. **5 Gráficas Representativas**: Visualizaciones de conceptos clave
5. **Exportación de Datos**: Resultados tabulados en CSV

---

## 🎯 Problema de Negocio

**Objetivo**: Predecir cuántos jugadores asistirán a un campo de golf en un día dado

**Variables Disponibles**:
- Temperatura (°C)
- Humedad (%)
- Velocidad del viento (km/h)
- Tipo de clima (Soleado, Nublado, Lluvioso)
- Número de jugadores (variable objetivo)

**Desafío Especial**: Solo 28 días de datos
- División: 14 días entrenamiento, 14 días prueba
- Limitación: Representa un caso realista con datos limitados
- Riesgo: Mayor probabilidad de overfitting

---

## 🤖 Modelos Implementados

### 1. **Regresión Lineal** [Underfitting]
```
Características: Alto Sesgo, Baja Varianza
├─ Complejidad: Muy Baja
├─ MSE Entrenamiento: 45.76
├─ MSE Prueba: 23.90
├─ Brecha: -21.86 (La prueba es mejor = sesgo alto)
├─ R² Prueba: 0.9456
└─ Interpretación: Modelo demasiado simple, no captura patrones
```

**¿Por qué funciona mejor en prueba?**
- El modelo es tan simple que no captura complejidad innecesaria
- No hay ruido que memorizar
- Error consistente en ambos conjuntos

---

### 2. **Polinomio Grado 2** [Balance Moderado]
```
Características: Sesgo Moderado, Varianza Moderada
├─ Complejidad: Media
├─ MSE Entrenamiento: ~0 (overfitting leve)
├─ MSE Prueba: 121.04
├─ Brecha: 121.04 (Diferencia significativa)
├─ R² Prueba: 0.7243
└─ Interpretación: Balance razonable para este dataset
```

**Comportamiento**:
- Se ajusta mejor a los datos que lineal
- Comienza a mostrar overfitting
- Aún generaliza razonablemente bien

---

### 3. **Polinomio Grado 3** [Sobreauste Moderado]
```
Características: Sesgo Bajo, Varianza Moderada-Alta
├─ Complejidad: Media-Alta
├─ MSE Entrenamiento: ~0 (overfitting)
├─ MSE Prueba: 193.98
├─ Brecha: 193.98 (Brecha muy alta)
├─ R² Prueba: 0.5582
└─ Interpretación: Comienza a memorizar datos
```

**Señal de Alerta**:
- La brecha train-test casi se triplica
- R² se reduce significativamente
- La complejidad adicional perjudica

---

### 4. **Polinomio Grado 5** [Overfitting Severo]
```
Características: Sesgo Muy Bajo, Varianza MUY ALTA
├─ Complejidad: Muy Alta
├─ MSE Entrenamiento: ~0 (ajuste perfecto)
├─ MSE Prueba: 1163.17
├─ Brecha: 1163.17 (¡Catastrófica!)
├─ R² Prueba: -1.6492 (¡Peor que predecir promedio!)
└─ Interpretación: Ejemplo de overfitting extremo
```

**Problema Crítico**:
- El modelo memoriza cada punto de entrenamiento
- Con solo 14 datos de entrenamiento, crear 21 términos polinomiales es excesivo
- Falla completamente en datos nuevos

---

### 5. **Árbol de Decisión** [Overfitting Extremo]
```
Características: Sesgo Muy Bajo, Varianza MUY ALTA
├─ Complejidad: Extremadamente Alta
├─ MSE Entrenamiento: 0 (ajuste perfecto)
├─ MSE Prueba: 408.07
├─ Brecha: 408.07 (Extrema)
├─ R² Prueba: 0.0706 (Prácticamente inútil)
└─ Interpretación: Divide el espacio en rectángulos arbitrarios
```

**Característica**:
- Los árboles sin restricción crean particiones para cada punto
- Con 14 datos, es trivial crear decisiones perfectas
- No hay capacidad de generalización

---

## 📊 Conceptos Fundamentales

### **¿QUÉ ES EL SESGO (BIAS)?**

El sesgo es el **error sistemático** causado por:
- Asumir el modelo es más simple que la realidad
- No capturar patrones complejos en los datos
- Resultado: **Underfitting**

**Síntomas de Alto Sesgo**:
- ✗ Error alto en conjunto de entrenamiento
- ✗ Error alto en conjunto de prueba
- ✗ Ambos errores similares (consistent, pero malo)
- ✗ Modelo no aprende patrones

**Ejemplo Visual**:
```
Realidad: Curva ondulante
Modelo Lineal: Línea recta
→ Siempre tiene error, no importa cuánto entrenes
```

---

### **¿QUÉ ES LA VARIANZA (VARIANCE)?**

La varianza es la **sensibilidad excesiva** a:
- Las fluctuaciones en datos de entrenamiento
- El ruido aleatorio en los datos
- Resultado: **Overfitting**

**Síntomas de Alta Varianza**:
- ✓ Error MUY bajo en entrenamiento
- ✗ Error ALTO en prueba
- ✗ Gran BRECHA entre train y test
- ✗ Pequeños cambios en datos → cambios grandes en predicciones
- ✗ Modelo memoriza, no generaliza

**Ejemplo Visual**:
```
Realidad: Curva ondulante suave
Modelo Polinomio Grado 10: Zigzag complicado
→ Ajusta perfectamente train, falla en test
```

---

### **EL TRADE-OFF SESGO-VARIANZA**

La ecuación fundamental:

$$\text{Error Total} = \text{Sesgo}^2 + \text{Varianza} + \text{Ruido}$$

**Donde**:
- **Sesgo²**: Distancia promedio del modelo verdadero
- **Varianza**: Variabilidad cuando cambias datos
- **Ruido**: Error irreducible en los datos

**Gráficamente**:

```
        Error
          ↑
   Underfitting │ Overfitting
        (Alto   │  (Alto
        Sesgo)  │ Varianza)
          │     │
      ┌───┼─────┴──────┐
      │   │      *     │  * = Error Total
      │   │    *   *   │
      │   │   *       * │
      │   │  *         *│ 
      │──→└────────────→│ Complejidad
        Punto Óptimo ▲
```

---

## 🔬 Análisis de Nuestros Datos

### Distribución de Clima
```
ENTRENAMIENTO (14 días):     PRUEBA (14 días):
├─ Soleado: 7                ├─ Soleado: 5
├─ Nublado: 3                ├─ Nublado: 7
└─ Lluvioso: 4               └─ Lluvioso: 2
```

**Problema**: Diferentes distribuciones = Desafío de generalización

### Correlaciones
```
Temperatura vs Jugadores:     +0.636 (correlación fuerte positiva)
Humedad vs Jugadores:         -0.478 (correlación moderada negativa)
Viento vs Jugadores:          -0.276 (correlación débil negativa)
```

**Implicación**: Las variables tienen relaciones complejas

---

## 📈 Resultados Comparativos

| Modelo | Complejidad | Sesgo | Varianza | MSE_Train | MSE_Test | Brecha | R²_Test | Conclusión |
|--------|-------------|-------|----------|-----------|----------|--------|---------|------------|
| **Lineal** | Baja | Alto | Baja | 45.76 | 23.90 | -21.86 | 0.946 | ✓ Buena generalización |
| **Polinomio 2** | Media | Moderado | Moderada | 0.00 | 121.04 | 121.04 | 0.724 | ⚠️ Balance aceptable |
| **Polinomio 3** | Media-Alta | Bajo | Moderada-Alta | 0.00 | 193.98 | 193.98 | 0.558 | ✗ Overfitting |
| **Polinomio 5** | Muy Alta | Muy Bajo | Muy Alta | 0.00 | 1163.17 | 1163.17 | -1.649 | ✗✗ Fallo total |
| **Árbol** | Muy Alta | Muy Bajo | Muy Alta | 0.00 | 408.07 | 408.07 | 0.071 | ✗ Inútil |

---

## 🎓 Lecciones Clave

### 1. **La Brecha train-test es crítica**
```
Brecha < 10      → ✓ Buen modelo
10 < Brecha < 50 → ⚠️ Aceptable pero con cuidado
Brecha > 100     → ✗ Problema serio de overfitting
```

### 2. **Datos limitados requieren modelos simples**
Con solo 28 datos:
- ✓ Modelos lineales o polinomios grado 2-3
- ✗ NO usar árboles profundos, redes neuronales complejas
- ✗ NO usar polinomios grado > 5

### 3. **El modelo "perfecto" en train es sospechoso**
Si MSE_train = 0 y 14 datos → Overfitting garantizado

### 4. **Underfitting es mejor que overfitting**
Un modelo simple que generaliza es mejor que uno complejo que no

### 5. **La validación es obligatoria**
Con datos pequeños, SIEMPRE:
- Separa train/test
- Considera validación cruzada
- Never overfit intentionally para ganar en train

---

## 💡 Recomendaciones Prácticas

### Para Este Problema Específico
1. **Usar Regresión Lineal o Polinomio Grado 2**
2. **Aplicar Regularización**: Ridge o Lasso para reducir varianza
3. **Validación Cruzada**: 5-fold o LOOCV para mejor estimación
4. **Recolectar Más Datos**: Aumentar de 28 a al menos 100 días

### Para Problemas con Datos Limitados
```
1. Feature Selection      → Eliminar variables irrelevantes
2. Regularización         → Ridge (L2), Lasso (L1)
3. Ensemble Methods       → Random Forest, Gradual Boosting
4. Transfer Learning      → Usar modelos pre-entrenados
5. Data Augmentation      → Generar datos sintéticos si es posible
6. k-Fold Cross-Validation → Validación robusta
7. Hold-out Validation    → Split train/val/test balanceado
```

---

## 📁 Archivos Generados

### Datos
- **golf_asistencia_28dias.csv**: Dataset de 28 días con 6 columnas
- **resultados_sesgo_varianza.csv**: Métricas de los 5 modelos

### Gráficas
1. **01_exploracion_datos.png**: Relación variables con asistencia
2. **02_sesgo_varianza_analisis.png**: Análisis principal (4 subgráficas)
3. **03_predicciones_reales_vs_predichas.png**: Scatter plots por modelo
4. **04_residuales_por_modelo.png**: Análisis de errores residuales
5. **05_curva_conceptual_sesgo_varianza.png**: Visualización teórica

### Notebook
- **sesgo_varianza_golf.ipynb**: Código completo ejecutable

---

## 🔗 Fórmulas Matemáticas Clave

### Error Cuadrático Medio (MSE)
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^2$$

### Coeficiente de Determinación (R²)
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_{i} - \hat{y}_{i})^2}{\sum(y_{i} - \bar{y})^2}$$

Interpretación:
- R² = 1: Predicción perfecta
- R² = 0: Predice al nivel del promedio
- R² < 0: Peor que predecir el promedio

### Raíz del MSE (RMSE)
$$RMSE = \sqrt{MSE}$$

Ventaja: En las mismas unidades que y (jugadores)

---

## 🎯 Conclusión

### El Punto Óptimo en Nuestro Caso

**Modelo Recomendado**: Regresión Lineal o Polinomio Grado 2

**Razones**:
1. ✓ Mejor balance entre sesgo y varianza
2. ✓ R² acept able (0.72-0.95)
3. ✓ Generaliza bien a datos nuevos
4. ✓ Interpretable y simple de explicar
5. ✓ No memoriza ruido del entrenamiento

### Lo Que NO Hacer
1. ✗ NO usar modelos complejos (polinomios grado > 3, árboles profundos)
2. ✗ NO optimizar solo para MSE_train
3. ✗ NO ignorar la brecha train-test
4. ✗ NO asumir que "más parámetros = mejor modelo"

### Pasos Siguientes (En Producción)
1. Recolectar más datos
2. Aplicar regularización
3. Implementar validación cruzada
4. Monitorear desempeño en datos reales
5. Reentrenar periódicamente

---

**Autor**: Sistema de Ciencia de Datos
**Fecha**: 2026-03-24
**Versión**: 1.0

---

## 📚 Referencias Teóricas

- Hastie, T., et al. (2009). "The Elements of Statistical Learning"
- James, G., et al. (2013). "An Introduction to Statistical Learning"
- Goodfellow, I., et al. (2016). "Deep Learning"
- Ng, A., & Jordan, M. (2006). "Bias-Variance Tradeoff"

