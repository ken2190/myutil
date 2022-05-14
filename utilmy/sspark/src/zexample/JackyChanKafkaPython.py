
from __future__ import print_function

import sys
import json 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

"""
Kafka with Python = Jacky Chan move simulation in Kafka Python , consume message from kafka broker and save in json file
"""

if __name__ == "__main__":
	    if len(sys.argv) != 3:
                print("Invalid args")
                exit(-1)

	    sc = SparkContext(appName="JackyChanKafkaWithPython")
	    ## 2 second
	    ssc = StreamingContext(sc, 2)

	    ##brokers: "192.168.59.103", broker.port: 9092
	    ##topic: "jackieChanCommand",
	    brokers, topic = sys.argv[1:]

		  # Define Kafka Consumer
	    kafkajackyMoves = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list": brokers})


	    ## run for each RDD in kafka Dstream and save it to json
	    kafkajackyMoves.foreachRDD(lambda jackyMoves: jackyMoves.wirte.mode(SaveMode.Append).json("jackieChanSimConfig.json"))

	    ssc.start()
	    ssc.awaitTermination()