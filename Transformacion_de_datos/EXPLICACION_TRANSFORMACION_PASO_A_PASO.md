# Explicacion Paso a Paso: Transformacion de Datos con Python y Scikit-learn

## 1. Objetivo de la actividad
En este ejercicio se prepara un conjunto de datos para entrenar un modelo de clasificacion.
La idea central es transformar correctamente los datos numericos y categoricos antes de modelar.

Archivos usados:
- Notebook: transformando_datos_01.ipynb
- Datos: data.csv

## 2. Librerias importadas y para que sirven
En el notebook se usan estas librerias:

- pandas: leer y manipular tablas de datos.
- numpy: operaciones numericas.
- SimpleImputer: rellenar valores faltantes.
- StandardScaler: estandarizar variables numericas.
- OneHotEncoder: convertir categorias en columnas binarias.
- ColumnTransformer: aplicar transformaciones distintas por tipo de columna.
- Pipeline: encadenar pasos de preprocesamiento y modelo.
- train_test_split: separar entrenamiento y prueba.
- LogisticRegression: modelo de clasificacion.
- accuracy_score y classification_report: evaluar rendimiento.

## 3. Carga del dataset
Se carga el archivo CSV con:

```python
data = pd.read_csv('data.csv')
```

Importante:
- El archivo data.csv debe estar en la misma carpeta que el notebook.
- La variable objetivo se llama target.

# target Ejemplo típico de generación de target (antes del notebook):

<!-- Regla de negocio: target = 1 si un cliente compra más de cierto umbral; si no, target = 0.
Etiquetado manual: un experto marca cada caso como clase 0 o 1. -->

## 4. Separar variables predictoras y variable objetivo
Se define:

```python
X = data.drop('target', axis=1)
y = data['target']
```

Interpretacion:
- X contiene las caracteristicas (edad, ingreso, comuna, etc.).
- y contiene la clase a predecir (0 o 1).

## 5. Identificar columnas numericas y categoricas
El codigo detecta automaticamente tipos de columnas:

```python
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
```

Esto evita escribir manualmente los nombres de columnas y facilita escalar el ejercicio a otros datasets.

## 6. Transformacion de datos numericos
Para columnas numericas se usa un pipeline:

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
```

Que hace:
- Reemplaza valores faltantes con la media.
- Estandariza (media 0 y desviacion 1) para que el modelo compare variables en una escala comun.

## 7. Transformacion de datos categoricos
Para columnas categoricas:

```python
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

Que hace:
- Rellena faltantes con la categoria mas frecuente.
- Convierte texto (por ejemplo comuna o membresia) en variables numericas binarias.
- handle_unknown='ignore' evita errores si aparece una categoria nueva en prueba.

## 8. Unir transformaciones con ColumnTransformer
Se combinan ambos pipelines:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

Ventaja:
- Cada tipo de dato recibe su transformacion correcta en una sola estructura.

## 9. Pipeline completo con modelo
Se integra preprocesamiento y clasificador:

```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

Beneficio didactico:
- Evita fugas de informacion (data leakage).
- El mismo flujo se aplica en entrenamiento y prediccion.

## 10. Entrenamiento y evaluacion
Proceso final:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

Interpretacion:
- train_test_split separa datos para medir generalizacion.
- fit aprende patrones en entrenamiento.
- predict estima clases en prueba.
- accuracy entrega porcentaje global de aciertos.
- classification_report muestra precision, recall y f1-score por clase.

## 11. Conceptos clave para explicar en clase
- Transformar datos es obligatorio antes de modelar.
- Numerico y categorico se tratan distinto.
- Un pipeline ordena y automatiza todo el flujo.
- La evaluacion debe hacerse con datos no vistos por el modelo.

## 12. Errores frecuentes y como evitarlos
- Error: no incluir la columna target.
  Solucion: verificar que exista en data.csv.

- Error: usar rutas incorrectas.
  Solucion: confirmar que notebook y CSV esten en la misma carpeta o ajustar la ruta.

- Error: no tratar nulos.
  Solucion: usar SimpleImputer como en el ejemplo.

- Error: codificar categorias manualmente y olvidar alguna.
  Solucion: usar OneHotEncoder(handle_unknown='ignore').

## 13. Sugerencia para continuar
Como siguiente practica, comparar LogisticRegression con otro modelo (por ejemplo RandomForestClassifier) usando el mismo preprocesador para analizar diferencias de rendimiento.
