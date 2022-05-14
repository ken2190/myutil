import pyspark
from pyspark.sql import SparkSession

conf = (
	pyspark.SparkConf()
      .set("spark.executor.instances", num_executors)
)
spark = (
	SparkSession.builder
	.appName("my_app_name")
	.config(conf=conf)
	.enableHiveSupport()
	.getOrCreate()
)