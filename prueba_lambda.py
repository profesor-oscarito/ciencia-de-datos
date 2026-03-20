from pyspark.sql import SparkSession # Importing SparkSession from the pyspark.sql module to create a Spark session for our batch job.
spark = SparkSession.builder.appName("BatchJob").getOrCreate() # Creating a Spark session with the name "BatchJob". This session will be used to 
# read data and perform transformations.

df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.show(5)