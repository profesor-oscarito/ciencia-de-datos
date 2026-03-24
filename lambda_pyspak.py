# COMENTA EN ESPAÑOL
import sys
import os

print("Python:", sys.executable)
print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

# Este código utiliza PySpark para realizar una agregación batch sobre un conjunto de datos de ventas históricas.
from pyspark.sql import SparkSession # Importamos la clase SparkSession para crear una sesión de Spark.
spark = SparkSession.builder.appName("LambdaBatch").getOrCreate() # Creamos una sesión de Spark con el nombre "LambdaBatch".

# Datos históricos
df_batch = spark.read.csv("datas.csv", header=True, inferSchema=True) # Leemos los datos históricos desde un archivo CSV.

# Agregación batch
batch_result = df_batch.groupBy("region").sum("gasto_mensual") 
# Agrupamos los datos por el campo "region" y sumamos el gasto mensual para cada región.

batch_result.show() # Mostramos el resultado de la agregación batch en la consola.