from pyspark.sql.functions import udf


get_timestamp_sec = udf(lambda x: int(x/1000), types.LongType())

rdd_df = rdd_df.withColumn('timestamp_sec', get_timestamp_sec(rdd_df['timestamp']))