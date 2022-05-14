import re
from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SimpleStreamingApp").getOrCreate()


def url2domain(url):
    url = re.sub('(http(s)*://)+', 'http://', url)
    parsed_url = urlparse(unquote(url.strip()))
    if parsed_url.scheme not in ['http','https']: return None
    netloc = re.search("(?:www\.)?(.*)", parsed_url.netloc).group(1)
    if netloc is not None: return str(netloc.encode('utf8')).strip()
    return None

@F.udf(returnType=ArrayType(StringType()))
def url_udf(urls_array):
   return [url2domain(url) for url in urls_array]

df = spark.read.json("hdfs:///user/ubuntu/l04/l04_train_merged_labels.json")
df = df.select(df.uid, df.visits.url.alias("urls"), df.gender_age)
df = df.withColumn("urls", url_udf(df.urls))

cv = CountVectorizer(inputCol="urls", outputCol="features")
indexer = StringIndexer(inputCol="gender_age", outputCol="label")
lr = LogisticRegression()
pipeline = Pipeline(stages=[cv, indexer, lr])

PIPE = pipeline.fit(df)
PIPE.save("hdfs:///user/ubuntu/l04/my_pipeline")
