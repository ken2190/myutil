###### Imports ######
import argparse
import os
from pyspark.sql.functions import udf
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
import pyspark
from pyspark import SparkFiles

###### Constants ######
CONTI_FEATURES  = ['year', 'bpm','nrgy', 'dnce', 'dB', 'live', 'val', 'dur','acous', 'spch', 'pop']
RESULTS_FILENAME_DEFAULT = 'results_top50_spotify.csv'

###### Arguments ######
parser = argparse.ArgumentParser(description='Top 50 songs most popular')
parser.add_argument('input_file', type=str, help='Input file path')
parser.add_argument('output_path', type=str, help='Output path to store results')
args = parser.parse_args()

###### Functions ######
# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 

if __name__=='__main__':

    # 0. Arguments
    filename = args.input_file
    results_path = args.output_path
    results_path_file = results_path + '/' + RESULTS_FILENAME_DEFAULT

    # 1. Spark Context
    sc = pyspark.SparkContext('local[*]')
    sqlContext = pyspark.SQLContext(sc)

    # 2. Read csv files
    df_spotify = sqlContext.read.csv(SparkFiles.get(filename), header=True, inferSchema= True)
    df_spotify = convertColumn(df_spotify, CONTI_FEATURES, FloatType())

    # 3. ETL / Computations
    df_spotify_results = df_spotify.groupBy("top genre", "country").agg({'pop': 'mean'}).sort("avg(pop)",ascending=False)

    # 4. Save results
    df_spotify_results.write.csv(results_path_file)

# Improvements: http://blog.appliedinformaticsinc.com/how-to-write-spark-applications-in-python/