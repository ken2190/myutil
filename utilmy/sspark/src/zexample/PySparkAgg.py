from pyspark.sql.types import *
import pyspark.sql.functions as F
import numpy as np

def find_median(values):
    try:
        median = np.median(values) #get the median of values in a list in each row
        return round(float(median),2)
    except Exception:
        return None #if there is anything wrong with the given values

# Code for Computing Median Aggregation
median_finder = F.udf(find_median,FloatType())

df2 = df.groupBy("id").agg(F.collect_list("num").alias("nums"))
df2 = df2.withColumn("median", median_finder("nums"))

# Code for Computing Min, Max, Avg Aggregation
df_min_max_avg = df.groupBy("id").agg(min(col("amount)).as("min_amount"), 
max(col("amount")).as("max_amount"),
avg(col("amount").as("avg_amount"))
)
