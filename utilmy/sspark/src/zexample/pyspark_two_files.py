  import sys 
  from pyspark.sql import SparkSession
  # Import data types
  from pyspark.sql.types import *
  from pyspark.sql.functions import when, lit, col, udf 
  
  
  spark = SparkSession.builder.appName("Python spark read two files").getOrCreate()
  
  accounts_file = sys.argv[1]
  data_file = sys.argv[2]
  
  account_df = spark.read.csv(accounts_file, header=True, inferSchema=True)
  data_df = spark.read.csv(data_file, header=True, inferSchema=True)
  
  result_df = account_df.join(data_df, "account_numbers")
  result_df.show()
  result_df.printSchema()