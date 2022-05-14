#!/usr/bin/env python
"""Extract events from kafka and write them to hdfs
"""
import json
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, from_json
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SQLContext
# extract question level information
def extract_info_level1(row):
    data = json.loads(row.value)
    result_list = []
    
    for flight_info in data["states"]:
        flight_info_row={"time":data["time"],
                        "icao24" : flight_info[0],
                        "callsign": flight_info[1],
                        "origin_country": flight_info[2],
                        "time_position": flight_info[3],
                        "last_contact":flight_info[4],
                        "longitude":flight_info[5],
                         "latitude":flight_info[6],
                         "baro_altitude":flight_info[7],
                         "on_ground":flight_info[8],
                         "velocity":flight_info[9],
                         "true_track":flight_info[10],
                         "vertical_rate":flight_info[11],
#                          "sensors":flight_info[12],
                         "geo_altitude":flight_info[13],
#                          "squawk":flight_info[14],
                         "spi":flight_info[15],
                         "position_source":flight_info[16]
                        }
        result_list.append(Row(**flight_info_row))
        
    return result_list
def main():
    """main
    """
    
    spark = SparkSession \
        .builder \
        .appName("ExtractEventsJob") \
        .getOrCreate()
    raw_events = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "planes") \
        .load()
    # save data in hadoop
    flight_status_info = raw_events \
        .select(raw_events.value.cast('string')) #\
        .rdd \
        .flatMap(extract_info_level1) \
        .toDF()
    
    write_action=flight_status_info \
        .writeStream \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/checkpoints_for_flight_status_info_table") \
        .option("path", "/tmp/flight_status_info_table") \
        .trigger(processingTime="10 seconds") \
        .start()
    write_action.awaitTermination()