#!/bin/python
from __future__ import print_function

# general packages
import sys
import config
import logging
import json
from random import choice

# pyspark streaming
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

# spark sql
from pyspark.sql import HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import desc
from pyspark.sql.functions import col
from pyspark.sql.functions import expr


# tweet_struct = StructType([
#     StructField('id', LongType(), False),
#     StructField('created_at', StringType(), False),
#     StructField('text', StringType(), False),
#     StructField('favorite_count', IntegerType(), True),
#     StructField('quote_count', IntegerType(), True),
#     StructField('retweet_count', IntegerType(), True),
#     StructField('reply_count', IntegerType(), True),
#     StructField('lang', StringType(), True),
#     StructField('coordinates', StringType(), True),
#     StructField('place', StringType(), True),
#     StructField('possibly_sensitive', StringType(), True),
#     StructField('user', MapType(StringType(), StringType()), False)
#     ])



def HandleJson(df):
    # fill na
    # check for possibly_sensitive
    # get rid of sensitive material
    if df.select("possibly_sensitive").show() == "true":
        return
    tweets = df.select("id",
        "created_at",
        expr('COALESCE(text, "null") AS text'),
        expr('COALESCE(favorite_count, 0) AS favorite_count'),
        expr('COALESCE(retweet_count, 0) AS retweet_count'),
        expr('COALESCE(quote_count, 0) AS quote_count'),
        expr('COALESCE(reply_count, 0) as reply_count'),
        expr('COALESCE(lang, "und") as lang'),
        expr('COALESCE(coordinates, 0) as coordinates'),
        expr('COALESCE(place, "null") as place'),
        col("user.id").alias("user_id"),
        expr("good_day() as date"),
        expr("rand_state() as state"),
        expr("rand_provider() as provider")
        # expr('concat("2018-02-", substring(created_at, 9, 2),  "T", substring(created_at,12,8), ".000") as datetime')
    )
    tweets.write.mode("append").insertInto("default.tweets")

    users = df.select("user.id",
        "user.name",
        "user.description",
        "user.followers_count",
        "user.location",
        "user.friends_count",
        "user.screen_name"
    )
    users.write.mode("append").insertInto("default.users")

def handleRDD(rdd):
    if not rdd:
        return
    try:
        # df=sqlContext.createDataFrame(json.loads(rdd.map(lambda x: x[1].encode('utf-8'))), samplingRatio=0.5)
        df=sqlContext.read.json(rdd.map(lambda x: x[1]))
        HandleJson(df)
    except Exception as ex:
        print(ex)

if __name__ == "__main__":

    sc = SparkContext("yarn", "TweetConsumer")
    ssc = StreamingContext(sc, 1)
    sqlContext = HiveContext(sc)
    # ssc.checkpoint("file:///" + getcwd())
    
    days = sqlContext.sql("select distinct date from transactions").collect()
    def good_day():
        return choice(days)[0].encode('utf-8')
    sqlContext.udf.register("good_day", good_day)

    states = sqlContext.sql("select State from us_states").collect()
    def rand_state():
        return choice(states)[0].encode('utf-8')
    sqlContext.udf.register("rand_state", rand_state)

    def rand_provider():
        return choice(['pandora', 'spotify'])
    sqlContext.udf.register("rand_provider", rand_provider)

    zkQuorum, topic = config.zkQuorum, config.topic
#    lines = KafkaUtils.createStream(ssc, [zkQuorum], "", [topic]
    lines = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list": zkQuorum})

    lines.foreachRDD(handleRDD)
    ssc.start()
    ssc.awaitTermination()

