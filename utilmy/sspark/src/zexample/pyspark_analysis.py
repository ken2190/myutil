#Data Analysis in PySpark
from pyspark.sql import SparkSession

spark = SparkSession.builder \
   .master("local[*]") \
   .appName("Spark Application : Reddit-Analysis") \
   .getOrCreate()

s3_folder = "s3://terality-public/datasets/reddit/medium/"

#Dataframe Creation/Loading Data
sp_comments=spark.read.parquet(f"{s3_folder}comments/").show(1)

#Data Sorting
comments_best_scores = sp_comments.sort_values(by="score", ascending=False)

#Dataframe Merging
sp_users=spark.read.parquet(f"{s3_folder}users.parquet").show(1)
sp_comments = sp_comments.join(sp_users,['author'],how='left')

#Dataframe Grouping/Aggregation
sp_comments.groupby("author")["ups"].sum().sort_values(ascending=False)

#Applying custom function (categorizing popularity)
popularity = udf(lambda x,y: popularity_func(x,y),StringType())
sp_comments.withColumn('reddit_popularity', popularity(col("ups"),col("score"))).show(1)