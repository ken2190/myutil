import findspark
#findspark.init()
import pyspark
import pandas as pd
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

sc = SparkContext()
spark = SparkSession \
    .builder \
    .appName("Python Spark DataFrames basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Spark read data file
#path = "VaporLinkReport.csv"
#df = spark.read.option("header", True).csv(path)
#df.show()
#df.printSchema()
#df.createTempView("h2s")
#df.select("H2SLevel").show()
#df.groupBy(hour("time").alias("hour")).agg(mean("H2SLevel").alias("mean")).sort(asc("hour")).show()

# Pandas read data file
data1 = pd.read_csv('VaporLinkReport.csv')
sdf = spark.createDataFrame(data1)
#sdf.printSchema()
sdf.createTempView("data")
#spark.sql("SELECT * FROM data").show()
#sdf.groupBy(hour("time").alias("hour")).agg(mean("H2SLevel").alias("mean")).sort(asc("hour")).show()

sdf2 = sdf.withColumn('GPM', sdf['dose']/3785)
#sdf2.printSchema()
#sdf2.groupBy(hour("time")).avg("GPM").sort(asc(hour("time"))).show()
sdf3 = sdf2.groupBy(hour("time")).agg(avg("H2SLevel"),avg("GPM")).sort(asc(hour("time")))

# Pandas Function
@pandas_udf("float")
def convert_dose(s: pd.Series) ->pd.Series:
    return s/3785
spark.udf.register("convert_dose", convert_dose)
spark.sql("SELECT hour(time), avg(convert_dose(dose)), avg(H2SLevel) FROM data GROUP BY hour(time) ORDER BY hour(time) ASC")

pd3 = sdf3.toPandas()
pd3.to_csv('data.csv',header=True, index=False)

