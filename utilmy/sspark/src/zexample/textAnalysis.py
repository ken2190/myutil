# By: Katie House
# Last Updated: 4/23/2018

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import *
import os
import tempfile


# Initiate Spark Session
spark = SparkSession.builder.master("local").appName("PS6").getOrCreate()
sc = spark.sparkContext

# Read Text files
u = spark.read.text("Ulysses.txt")
b = spark.read.text("Bible.txt")

# Clean data of punctuation
u = u.withColumn('value' , lower(regexp_replace('value', '[^0-9A-Za-z:\s]', '')))
b = b.withColumn('value' , lower(regexp_replace('value', '[^0-9A-Za-z:\s]', '')))

# Split lines and explode
uDF = u.select(split(u.value, " ").alias('word'))
bDF = b.select(split(b.value, " ").alias('word'))

uDF = uDF.select(explode('word')).withColumnRenamed("col", "word")
bDF = bDF.select(explode('word')).withColumnRenamed("col", "word")


# Remove null values
uDF = uDF.dropna()
bDF = bDF.dropna()

# Eliminate all verse numbers    
bDF = bDF.filter("word NOT LIKE '__:___:___'")

# Eliminate Stop Words
stopwords = list(sc.textFile('stopwords.txt').collect())
uDF = uDF.filter(~uDF.word.isin(stopwords))
bDF = bDF.filter(~bDF.word.isin(stopwords))

# Output results
bDF = bDF.groupBy("word").count().withColumnRenamed("count", "bcount")
uDF = uDF.groupBy("word").count().withColumnRenamed("count", "ucount")

jDF = uDF.join(bDF, 'word')
jDF.toPandas().to_csv('book_data.csv')

