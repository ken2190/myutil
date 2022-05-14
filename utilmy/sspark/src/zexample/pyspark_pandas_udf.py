from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

def plus_one_func(v):
    return v + 1

plus_one = pandas_udf(plus_one_func, returnType=IntegerType())

spark = SparkSession.builder.appName("pyspark_pandas_udf").getOrCreate()
df = spark.read.load("/HiBench/DataFrame/Input")
df = df.withColumn('count', plus_one(df["count"]))
df.write.format("parquet").save("/HiBench/DataFrame/Output")