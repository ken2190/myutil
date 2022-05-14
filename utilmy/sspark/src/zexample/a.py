'''
Created on 25-Jun-2020

@author: srinivasan
'''
import json

from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType

spark = SparkSession.builder.appName("firstSample")\
    .master("local").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "10")


def row_toDict(row,
               order_cols=['id', 'type']):
    data = json.loads(row)
    return {data[order_cols[0]]:
            data[order_cols[-1]]}


res = spark.read.json('entity.json')\
    .select("RES.*").filter("type is not NULL")\
    .dropDuplicates(['id'])\
    .toJSON().map(row_toDict).collect()
    
aqs_msgs = {k: v for d in res for k, v in d.items()}

get_event = udf(lambda x: aqs_msgs.get(x, None),
               StringType())
spark.read.json('people.json')\
    .select("RES.*").withColumn('event_type',
                                 get_event('id'))\
                                 .dropDuplicates().show()



