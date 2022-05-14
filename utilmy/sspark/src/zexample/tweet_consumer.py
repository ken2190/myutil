#!/bin/python
from __future__ import print_function
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
# from pyspark.sql import SQLContext
from pyspark.sql.functions import desc
from pyspark.sql import HiveContext
from os import getcwd
import logging
from pyspark.sql.functions import col

sc = SparkContext("local[2]", "TweetConsumer")
ssc = StreamingContext(sc, 10)
sqlContext = HiveContext(sc)
# ssc.checkpoint("file:///" + getcwd())
socket_stream = ssc.socketTextStream("127.0.0.1", 5555)
lines = socket_stream.window(20)

def HandleJson(df):
    # fill na 
    # check for possibly_sensitive
    # get rid of sensitive material
    if df.select("possibly_sensitive").show() == "true":
        return
    tweets = df.select("id",
        "created_at",
        "text",
        "favorite_count",
        "retweet_count",
        "quote_count",
        "reply_count",
        "lang",
        "coordinates",
        "place",
        col("user.id").alias("user_id")
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
    df=sqlContext.read.json(rdd)
    try:
        HandleJson(df)
    except Exception as ex:
        print(ex)

lines.foreachRDD(handleRDD)
ssc.start()
ssc.awaitTermination()
