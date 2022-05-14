from pyspark.sql.functions import monotonically_increasing_id as mi
id=mi()
df1 = df1.withColumn("match_id", id)