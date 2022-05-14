from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf

from pyspark.sql import Row

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import csv
import hashlib

def readBuildings(path):
    df = spark.read.csv('/data/drio/buildings.map.csv')
    d = {}
    first = True
    for row in df.rdd.collect():
        if first:
            first = False
        else:
            id, name = row
            d[name] = int(id)
    return d

def doStep1(df, buildings):
    def genSHA(val):
        return hashlib.sha256(val.encode('utf-8')).hexdigest()[0:12]

    def genBuildingID(name):
        # FIXME: global buildings
        if name in buildings:
            return buildings[name]
        return -1

    def genUpdateTS(updateTS):
        return int(updateTS/1000)

    df_extras = df.withColumn('campusName', F.trim(F.split('location', '>')[0]))
    df_extras = df_extras.withColumn('update_minute', df_extras.updateHMS.substr(3, 2))

    udfGenUpdateTS = udf(genUpdateTS, IntegerType())
    df_extras = df_extras.withColumn('updateTS', udfGenUpdateTS("updateTime"))

    udfGenSHA = udf(genSHA, StringType())
    df_extras = df_extras.withColumn('sha', udfGenSHA("macAddress"))

    udfGenBuildingID = udf(genBuildingID, IntegerType())
    df_extras = df_extras.withColumn('buildingID', udfGenBuildingID("buildingName"))

    df_only = df_extras.filter((df_extras.campusName.startswith('Medford')) | (df_extras.campusName == 'SMFA'))
    df_only = df_only.filter(df_only.status == 'ASSOCIATED')

    df_sel = df_only.select('sha', 'buildingID', 'updateTS')

    return df_sel

# Main
#######
sc = SparkContext('local', 'drio-wifi')
spark = SparkSession(sc)

yearMonth = '202011'
df = spark.read.parquet(f'/data/network_data/wifi_converted/{yearMonth}')
df = df.cache()

buildings = readBuildings('/data/drio/buildings.map.csv')
df_sel = doStep1(df, buildings)
df_sel.coalesce(1).write.csv(f'/data/drio/wifi.{yearMonth}', mode="overwrite", header=True)
