

from __future__ import print_function
import sys
import json
from pyspark import SparkContext
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql.functions import col

"""
Spark with Python = Jacky Chan move simulation in Spark Python , process json file and analyse moves
"""

if __name__ == "__main__":
  
    spark = SparkSession.builder.master("local").appName("JackyChanMoves").getOrCreate()


    ## Create dataframe from json file with	timestamp,style,action,weapon,target,strength
    dfJackyAllMoves = spark.read.json("jackieChanSimConfig.json")	

    ## Filter out  'Block' and 'JUMP' 
    dfJackyMoves=dfJackyAllMoves.where(col("action").isin(["PUNCH", "KICK"]))

    ## calculate remaining strength
    def strenghRemain(strength,target):
        target_power = {'HEAD': 20, 'ARMS': 50,'BODY': 30,'LEGS': 50 }
        remainingStrg = strength - target_power[target] 
        return remainingStrg

    ## UDF 
    udfstrenghRemain=udf(strenghRemain, DoubleType())

    ##add column strengthRemaining using UDF
    dfJackyMoveswithstrength=dfJackyMoves.withColumn("strengthRemaining", udfstrenghRemain("strength","target")) 

    ## Ans 1 = Jacky's favorite style
    dffavStyle=dfJackyMoveswithstrength.groupBy("style").agg(max("style"))

    ## Ans 2 = Jacky's killing blow where remaining streangth is <= 0 , assuming when strength is <= zero , simulation stop
    dfKillingBlow=dfJackyMoveswithstrength.where($"strengthRemaining" <= 0).sort(asc("timestamp")).take(1)


    spark.stop()