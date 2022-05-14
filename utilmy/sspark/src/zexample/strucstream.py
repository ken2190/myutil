from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import split
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

import numpy as np

# Padding allows for considerdation of log time stamp issues,
# network delay, etc.
padding = 1.0

# Minimum data points to consider activity periodic
min_dp = 5

# Minimum score to be deemed periodic
p_criteria = 0.9

# Periodicity detection
def periodicity(df):
	if len(df) <= min_dp:
		return 0.0
	time_df = sorted([int(val) for val in df])
	time_diff = [round((time_df[i] - time_df[i-1]) / padding) * padding for i in range(1,len(time_df))]
	values, counts = np.unique(time_diff, return_counts=True)
	probs = counts/len(time_diff)
	en = sum([-prob * np.log(prob) for prob in probs]) / np.log(len(time_diff)) 
	entropy = 1.0 - en 
	return float(entropy)

# Only retain last 25 timestamps
def filter_tstamp(df):
	if len(df) > 25:
		return df[-25:]
	else:
		return df


tstamp_udf = udf(filter_tstamp)
periodic_udf = udf(periodicity, FloatType())


spark = SparkSession \
	.builder \
	.appName("StructuredPeriodicity") \
	.getOrCreate()


lines = spark \
	.readStream \
	.format("socket") \
	.option("host", "localhost") \
	.option("port", 9999) \
	.load()

# Split the line into values
split_col = split(lines.value, ",")

n_log = lines.withColumn("source_ip", split_col.getItem(0)) \
	.withColumn("destination_ip", split_col.getItem(1)) \
	.withColumn("timestamp", split_col.getItem(2)) \
	.drop("value")

# Generate arrays of interstitial times
p_log = n_log.groupBy("source_ip","destination_ip") \
	.agg(collect_list("timestamp").alias("timestamp")) \
	.select("source_ip", "destination_ip", "timestamp", tstamp_udf("timestamp").alias("timestamp_new")) \
	.select("source_ip", "destination_ip", "timestamp", periodic_udf("timestamp_new").alias("entropy"))

# Start running the query
query = p_log \
	.writeStream \
	.outputMode("complete") \
	.format("console") \
	.start()

query.awaitTermination()