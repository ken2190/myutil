"""
python3

Tools:

A)
https://stackoverflow.com/questions/24678308/how-to-find-location-with-ip-address-in-python
https://github.com/ip2location/IP2Location-Python
https://www.ip2location.com/development-libraries

or

B)
https://pythonhosted.org/python-geoip/
https://dev.maxmind.com/geoip/geoip2/geolite2/


Paper:
https://science.sciencemag.org/content/353/6304/1151.full?ijkey=7Wq4RKNGjbIvw&keytype=ref&siteid=sci
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Y3VPIG

"""

#!hdfs dfs -mkdir /user/smgoodman/internet_penetration
#!hdfs dfs -put /home/cdsw/routed_all.txt /user/smgoodman/internet_penetration
#!hdfs dfs -put /home/cdsw/IP2LOCATION-LITE-DB5.IPV6.BIN /user/smgoodman/internet_penetration/IP2LOCATION-LITE-DB5.IPV6.BIN

from pyspark.sql import SparkSession
from pyspark import SparkFiles

#ip_bin_path = "file:///home/cdsw/IP2LOCATION-LITE-DB5.IPV6.BIN"
#ip_bin_path = "/user/smgoodman/internet_penetration/IP2LOCATION-LITE-DB5.IPV6.BIN"
#ip_bin_path = "/home/cdsw/IP2LOCATION-LITE-DB5.IPV6.BIN"
#ip_bin_path = "file:///home/cdsw/IP2LOCATION-LITE-DB5.IPV6.BIN"
ip_bin_path = "hdfs:///user/smgoodman/internet_penetration/IP2LOCATION-LITE-DB5.IPV6.BIN"

spark = SparkSession.builder.appName("iptest").master("yarn").getOrCreate()

spark.sparkContext.addPyFile("/home/cdsw/IP2Location.py")
spark.sparkContext.addFile(ip_bin_path)

from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

import IP2Location


#hdfs path
routed_path = "/user/smgoodman/internet_penetration/routed_all.txt"

spark_df = spark.read.format('csv').options(header='false', inferSchema='true').load(routed_path)

spark_df = spark_df.select(col("_c0").alias("ip"), col("_c1").alias("year"))


spark_df.cache()

spark_df.printSchema()

#spark_df.show()



def get_lon(ip):
  ip_bin = SparkFiles.get(ip_bin_path)
  db = IP2Location.IP2Location(ip_bin)
  return db.get_longitude(ip)

def get_lat(ip):
  ip_bin = SparkFiles.get(ip_bin_path)
  db = IP2Location.IP2Location(ip_bin)
  return db.get_latitude(ip)


lon_udf = udf(get_lon, FloatType())
lat_udf = udf(get_lat, FloatType())


spark_geo_df = spark_df.withColumn('lon', lon_udf('ip'))
spark_geo_df = spark_geo_df.withColumn('lat', lat_udf('ip'))
spark_geo_df.printSchema()

spark_geo_df.head()
