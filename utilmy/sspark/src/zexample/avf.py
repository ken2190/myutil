from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import udf, array, count
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.getOrCreate()

# assumptions: all columns in data are categorical
df = spark.read.option('inferSchema', 'True').csv(data), header=True)

for col in df.columns:
    # get counts for each column and replace values in df[col]
    df = df.withColumn(col, count(col).over(Window.partitionBy(col)))

# function to average the frequencies by row
avg_cols = udf(lambda array: sum(array) / len(array), DoubleType())

# add column 'avf' to df
df = df.withColumn('avf', avg_cols(array(*df.columns)))

df.show()