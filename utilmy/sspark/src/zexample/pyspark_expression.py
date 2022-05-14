from pyspark.sql import SparkSession

def plus_one_func(v):
    return v + 1

spark = SparkSession.builder.appName("pyspark_expression").getOrCreate()
df = spark.read.load("/HiBench/DataFrame/Input")
df = df.withColumn('count', plus_one(df["count"]))
df.write.format("parquet").save("/HiBench/DataFrame/Output")