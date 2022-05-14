from pyspark.sql.types import ArrayType, IntegerType
from urllib.parse import urlparse
import re

def process_url(dataframe, spark_session, url_column) -> DataFrame:

   create_features = udf(
       f=lambda url: create_query_based_featuers(url),
       returnType=ArrayType(elementType=IntegerType()),
   )
   spark_session.udf.register("create_features", create_features)

   def create_query_based_featuers(url):
       query = urlparse(url).query
       # check if there is a query in the string; count how many '='; count how many numbers
       return (
           int(bool(query)),
           len(re.findall("=", query)),
           len(re.findall("\d+", query)),
       )

   dataframe = (
       dataframe.withColumn("all_features", create_features(url_column))
       .cache()
       .withColumn("is_query", col("all_features").getItem(0))
       .withColumn("num_equals", col("all_features").getItem(1))
       .withColumn("num_numbers", col("all_features").getItem(2))
       .drop("all_features")
   )

   return dataframe