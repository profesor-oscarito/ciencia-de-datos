from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LambdaBatch").getOrCreate()

# Datos históricos
df_batch = spark.read.csv("ventas_historicas.csv", header=True, inferSchema=True)

# Agregación batch
batch_result = df_batch.groupBy("Producto").sum("Cantidad")

batch_result.show()