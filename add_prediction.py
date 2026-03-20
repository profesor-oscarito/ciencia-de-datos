
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
df = pd.read_csv('datos_crun.csv')

# Preprocesamiento de datos
# Convertir variables categóricas a variables ficticias/indicadoras
df = pd.get_dummies(df, drop_first=True)

# Dividir los datos en características (X) y variable objetivo (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de Regresión Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

# Predecir sobre todo el conjunto de datos
predictions = model.predict(X)

# Agregar la columna de predicción al DataFrame
df['prediccion_churn'] = predictions

# Guardar el DataFrame actualizado en el mismo archivo CSV
df.to_csv('datos_crun.csv', index=False)

print("Se ha agregado la columna 'prediccion_churn' al archivo 'datos_crun.csv'.")
