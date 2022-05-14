from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml import clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sent_emb

# Create a Spark Session
spark = SparkSession.builder \
    .appName("SparkML Clustering") \
    .config("spark.executor.memory", "45g") \
    .config("spark.yarn.executor.memoryOverhead", "45g") \
    .getOrCreate()

# Add the necessary files to the Spark Context
spark.sparkContext.addFile('cc.id.300.bin')
spark.sparkContext.addPyFile('sent_emb.py')

# Load the Data
# If Data is small then use Pandas to load data first
# pandas_df = pd.read_csv("https://github.com/bgweber/Twitch/raw/master/Recommendations/games-expand.csv")
# spark_df = spark.createDataFrame(pandas_df)

csv_file = 'gs://leo-gc-sandbox/raw_bq_extract/data_*.csv'
  
schema = StructType([
    StructField("product_id",IntegerType()),
    StructField("product_name",StringType()),
    StructField("product_description",StringType()),
    StructField("category_id",IntegerType()),
    StructField("product_price",DoubleType())
])
        
df = spark.read.csv(csv_file, schema, header="true", escape='"', multiLine=True)
df = df.dropna(how='any')
df = df.withColumn("product_name", f.lower(df.product_name))

# Convert the text into a Spark VectorUDT Embedding Vector for SparkML can work
udf_sent_emb = f.udf(lambda name: Vectors.dense(sent_emb.sent_emb(name)), VectorUDT())

# Create the sentence embedding
df = df.withColumn("sentence_emb_dense", udf_sent_emb(df["product_name"]))

# Spark ML Clustering
# Reference: https://rsandstroem.github.io/sparkkmeans.html
cost = []
k_clusters = [50000, 60000]

for k in k_clusters:
    kmeans = clustering.KMeans(initMode='random').setK(k).setSeed(1).setFeaturesCol("sentence_emb_dense")
    model = kmeans.fit(df)
    cost.append(model.computeCost(df))
    model.save('gs://leo-gc-sandbox/notebooks/jupyter/models/kmeans_'+str(k))

print(cost)