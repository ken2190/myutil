from pyspark.sql.functions import monotonicallyIncreasingId

# This will return a new DF with all the columns + id
res = df.withColumn("id", monotonicallyIncreasingId())