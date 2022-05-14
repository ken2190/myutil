import os
import sys
import subprocess

os.environ["SPARK_HOME"] = r"/usr/lib/spark"

# Set PYTHONPATH for Spark
for path in [r'/usr/lib/spark/python/', r'/usr/lib/spark/python/lib/py4j-src.zip']:
    sys.path.append(path)

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import *

sc = SparkContext()
sqlContext = SQLContext(sc)

def valueToCategory(value):
   if   value == '2': return 'unconfirmed'
   elif value == '3': return 'confirmed'
   else: return '-9999'

df = sqlContext.read.csv('output')

# filter
filtered = df.filter(df['_c3'] > 2015)

# add stupid confidence_text
# http://stackoverflow.com/a/37263999/4355916
udfValueToCategory = udf(valueToCategory, StringType())
df_with_cat = filtered.withColumn("category", udfValueToCategory("_c2"))

# rename columns
# oldColumns = df_with_cat.columns
# newColumns = ['lon', 'lat', 'confidence', 'year', 'julian_day', 'area', 'emissions', 'climate_mask', 'iso', 'adm1', 'adm2', 'confidence_text']
# df_with_cat = reduce(lambda df_with_cat, idx: df_with_cat.withColumnRenamed(oldColumns[idx], newColumns[idx]), xrange(len(oldColumns)), df_with_cat)

s3_temp_dir = r's3://gfw2-data/alerts-tsv/temp/output-glad2-all-20170509-2'
df_with_cat.write.csv(s3_temp_dir)

# s3_cmd = ['s3-dist-cp', '--src', r'hdfs:///processed/', '--dest', r's3://gfw2-data/alerts-tsv/temp/output-glad2-summary-v5/', '--groupBy', '".*(part-r*).*"']

s3_cmd = ['s3-dist-cp', '--src', s3_temp_dir, '--dest', 's3://gfw2-data/alerts-tsv/temp/output-glad2-summary-v9/', '--groupBy', '.*(part-r*).*']
subprocess.check_call(s3_cmd)

remove_temp_cmd = ['aws', 's3', 'rm', s3_temp_dir, '--recursive']

subprocess.check_call(remove_temp_cmd)
