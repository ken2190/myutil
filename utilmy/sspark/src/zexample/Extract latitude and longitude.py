#Import libraries
import pyproj
import pandas as pd

#Pyspark functions - ONLY - DO not import functions like "import * from <package name>"
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType


#Read dataframe with XY(eastings and northings) and population value
pop_df = spark.read.csv("XY_POP_DATA_FULL.csv", header='True', inferSchema='True')

#Make required casting 
pop_df = pop_df.withColumn("X", col("X").cast(DoubleType()))
pop_df = pop_df.withColumn("Y", col("Y").cast(DoubleType()))
pop_df = pop_df.withColumn("Population", col("Population").cast(DoubleType()))

#Create a CRS projection string with same value as of the original GHS POP File
p = pyproj.proj.Proj(r'+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs')

#Function to get latitude and longitude from XY Coords
def get_lat(x,y):
    return p(x,y,inverse=True)[1]
    
#Function to get latitude and longitude from XY Coords
def get_lon(x,y):
    return p(x,y,inverse=True)[0]

#Convert functions to spark UDFs
spark_udf_lat = udf(get_lat, DoubleType())
spark_udf_lon= udf(get_lon, DoubleType())

#Apply on the population XY DF
pop_df = pop_df.withColumn("latitude", spark_udf_lat(col("X"),col("Y")))
pop_df = pop_df.withColumn("longitude", spark_udf_lon(col("X"),col("Y")))