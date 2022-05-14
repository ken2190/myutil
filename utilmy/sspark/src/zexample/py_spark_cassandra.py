
"""

wget https://www.apache.org/dyn/closer.lua/spark/spark-2.3.0/spark-2.3.0-bin-hadoop2.7.tgz

setx SPARK_HOME C:\spark\spark-2.3.0-bin-hadoop2.7

setx HADOOP_HOME C:\spark\spark-2.3.0-bin-hadoop2.7

setx PYSPARK_DRIVER_PYTHON ipython

setx PYSPARK_DRIVER_PYTHON_OPTS notebook

Add ;C:\spark\spark-2.3.0-bin-hadoop2.7\bin to your path.

"""

from datetime import datetime
# create data folder to download stock data to
import os

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')

if not os.path.exists(data_dir):
    os.makedirs('data')

# pip install pandas
import pandas as pd

keyspace = "ticker_plant"
table = 'datapoint'

# pyspark --master local[2]
# pip install pyspark
# pip install findspark

import findspark
findspark.init()
from pyspark.sql.functions import min, max, col, lag, when, isnull, mean, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as func
# only run after findspark.init()
import pyspark
import pyspark.sql.functions as f
from pyspark.sql import SparkSession, Row, Column, SQLContext
from pyspark.sql.functions import pandas_udf, PandasUDFType, udf,  datediff, to_date, lit
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType, IntegerType, StringType, DateType, TimestampType, FloatType
from pyspark.sql import Window
from pyspark import SparkContext
# spark = SparkContext()
# sc = pyspark.SparkContext('local[3]')

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
import pyspark_cassandra
import pyspark_cassandra
from pyspark.context import SparkConf
from pyspark_cassandra import CassandraSparkContext, saveToCassandra

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

conf = SparkConf().set("spark.cassandra.connection.host", "localhost").setMaster("local[*]").setAppName("PySpark Cassandra Driver")
sc = SparkContext(conf=conf)
spark = sc.getOrCreate()

tableRDD = sc.cassandraTable(keyspace, table)
sqlContext = SQLContext(sc)

sqlContext.read.format("org.apache.spark.sql.cassandra").options(table=table, keyspace=keyspace).load().show()

ds_people = (
    spark.read.format('org.apache.spark.sql.cassandra').options(table=table, keyspace=keyspace).load()
)
conf = SparkConf()

sc = CassandraSparkContext(conf=conf)

df = (spark
    .read
    .format("org.apache.spark.sql.cassandra")
    .options(table=table, keyspace=keyspace)
    .load())

sdf.show()
sdf.printSchema()
sdf.select('date','close').show(5)
sdf.write.saveAsTable("aapl")

w = Window.partitionBy().orderBy("date")

sdf = sdf.sort('date', ascending=True)
sdf.show()

sdf = sdf.withColumn("prevClose", lag(sdf.adjClose).over(w))
formula_pct_chg = (((sdf.adjClose - sdf.prevClose)/sdf.prevClose)*100)
sdf = sdf.withColumn("pct_chg", when(isnull(formula_pct_chg), 0).otherwise(formula_pct_chg))

sdf.show()

scaled_result = ((col("adjClose") - col("prevClose")) / (col("prevClose")))*100
sdf = sdf.withColumn("pct_chg", when(isnull(formula_pct_chg), 0).otherwise(formula_pct_chg))

scaled_result.show()

scaled_result = (col("adjClose") - min("adjClose").over(w)) / (max("adjClose").over(w) - min("adjClose").over(w))*100
sdf = sdf.withColumn("pct_chg2", scaled_result)

sdf.show()
oilPriceDatedDF = sdf.withColumn("date", sdf("date"))

windowDF = sdf.groupBy(w(sdf.col("date"),"1 week", "1 week", "4 days"))


df_lag = sdf.withColumn('prev_day_price',func.lag(sdf['adjClose']).over(Window.partitionBy("date")))

result = df_lag.withColumn('daily_return',(df_lag['price'] - df_lag['prev_day_price']) / df_lag['price'])

## Force the output to be float
def square_float(x):
    return float(x**2)

square_udf_float2 = udf(lambda z: square_float(z), FloatType())
# Integer type output

df_result = (
    sdf.select('Open',
              'Close',
               square_udf_float2('Open').alias('Open_squared'),
               square_udf_float2('Close').alias('Close_squared'))
)

df_result.show()


w = Window().orderBy("date")
w = Window.partitionBy().orderBy("date")
t = sdf.withColumn("percentDiff", (col("close") - lag("close", 1).over(w)) / lag("close", 1).over(w)).groupBy("date").agg(mean("percentDiff"))
t.show()

