# example with pandas UDF (series -> series). for more examples see below
# https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html
# https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html
# pyspark.sql.functions.pandas_udf
# pyarrow required (arrow used to transfer data to/from Pandas)
# this is only an example, it will probably not offer significant performance gains

# set environment variables
import os
os.environ['PYSPARK_PYTHON'] = r'D:\learning\2021_07_03_learningSparkV2Local\venv\Scripts\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'D:\learning\2021_07_03_learningSparkV2Local\venv\Scripts\python.exe'
os.environ['JAVA_HOME'] = r'D:\Applications\java\jdk1.8.0_172'
# https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems
os.environ['HADOOP_HOME'] = r'D:\learning\2021_07_03_learningSparkV2Local\hadoop-3.2.1'

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
import pandas as pd
from time import perf_counter

# get a spark session
spark = SparkSession.builder.appName('learn').getOrCreate()

# create a dummy dataframe to experiment with, this creates one column called id
df = spark.range(100_000_000)

# set up a pandas UDF (type hints are used in spark 3.x)
def cube(x: pd.Series) -> pd.Series:
    return x**3
cube_udf = f.pandas_udf(cube, LongType())
# alternative syntax using decorators
# @f.pandas_udf(LongType())
# def cube(x: pd.Series) -> pd.Series:
#     return x**3

# apply the udf
start = perf_counter()
res = df.select(f.col('id'), cube_udf(f.col('id')))
res.count()
print(f'Pandas UDF took {perf_counter() - start} seconds')

# compare with a standard python UDF (slow communication between JVM and python)
def cube_slow(x: int) -> int:
    return x**3
cube_slow_udf = f.udf(cube_slow, LongType())

# apply the slow udf
start = perf_counter()
res = df.select(f.col('id'), cube_slow_udf(f.col('id')))
res.count()
print(f'Python UDF took {perf_counter() - start} seconds')
