from pyspark import SparkContext, SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import DataFrame, udf, col
from geoip2 import database
from geoip2.errors import AddressNotFoundError
from geoip2.models import City

sc = SparkContext()
spark = SparkSession(sc)

sc.addFile("hdfs://path/to/GeoLite2-City.mmdb")  # or another geo db & can by used http(s)://, etc instead of hdfs://

df = spark.read.json("/path/to/data")  # or csv, avro, etc.

schema = StructType([
    StructField("country", StringType(), True),
    StructField("region", StringType(), True),
])

@udf(returnType=schema)
def geoip(ip):
    geo = database.Reader(SparkFiles.get("GeoLite2-City.mmdb"))
    
    try:
        result = geo.city(ip)
        pass
    except AddressNotFoundError:
        return {"country": None, "region": None}
    
    specific = result.subdivisions
    # [0] is for "Kraj" in Czech Republic, because python geoip2 does not have `subdivisions.least_specific` as in Java lib
    # http://maxmind.github.io/GeoIP2-java/doc/v2.6.0/com/maxmind/geoip2/model/AbstractCityResponse.html#getLeastSpecificSubdivision--
    return {"country": result.country.name, "region": specific[0].name if specific else None} 

df.withColumn("geoip", geoip("ip")).select("ip", "geoip.*").show()