from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType, DoubleType, BooleanType
import math
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType
import numpy as np
import random
from pyspark.sql import Row
from pyspark.sql import functions as F
import functools
from pyspark.sql.functions import col, rand

randomFloatsDF = (spark.range(0, 100 * 1000 * 1000)
  .withColumn("id", (col("id") / 1000).cast("integer"))
  .withColumn("random_float", rand())
)

randomFloatsDF.cache()
randomFloatsDF.count()

display(randomFloatsDF)







# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd


# %%
get_ipython().run_line_magic('run', '"./Includes/Classroom-Setup"')


# %%
path = "/mnt/training/EDGAR-Log-20170329/EDGAR-Log-20170329.csv"

logDF = (spark
  .read
  .option("header", True)
  .csv(path)
  .sample(withReplacement=False, fraction=0.3, seed=3) # using a sample to reduce data size
)

display(logDF)


# %%
from pyspark.sql.functions import col

serverErrorDF = (logDF
  .filter((col("code") >= 500) & (col("code") < 600))
  .select("date", "time", "extention", "code")
)

display(serverErrorDF)


# %%
from pyspark.sql.functions import from_utc_timestamp, hour, col

countsDF = (serverErrorDF
  .select(hour(from_utc_timestamp(col("time"), "GMT")).alias("hour"))
  .groupBy("hour")
  .count()
  .orderBy("hour")
)

display(countsDF)


# %%
workingDir = '.'


# %%
targetPath = workingDir + "/serverErrorDF.parquet"

(serverErrorDF
  .write
  .mode("overwrite") # overwrites a file if it already exists
  .parquet(targetPath)
)


# %%



# %%
# TODO

wikiDf = (spark.read
  .option("delimiter", "\t")
  .option("header", True)
  .option("timestampFormat", "mm/dd/yyyy hh:mm:ss a")
  .option("inferSchema", True)
  .csv("/mnt/training/wikipedia/pageviews/pageviews_by_second.tsv")
)
display(wikiDf)


# %%
get_ipython().run_line_magic('scala', '')
(/, run, this, regardless, of, language, type)
Class.forName("org.postgresql.Driver")


# %%
jdbcHostname = "server1.databricks.training"
jdbcPort = 5432
jdbcDatabase = "training"

jdbcUrl = f"jdbc:postgresql://{jdbcHostname}:{jdbcPort}/{jdbcDatabase}"


# %%
connectionProps = {
  "user": "readonly",
  "password": "readonly"
}


# %%
tableName = "training.people_1m"

peopleDF = spark.read.jdbc(url=jdbcUrl, table=tableName, properties=connectionProps)

display(peopleDF)


# %%
integerDF = spark.range(1000, 10000)

display(integerDF)


# %%
from pyspark.sql.functions import col, max, min

colMin = integerDF.select(min("id")).first()[0]
colMax = integerDF.select(max("id")).first()[0]

normalizedIntegerDF = (integerDF
  .withColumn("normalizedValue", (col("id") - colMin) / (colMax - colMin) )
)

display(normalizedIntegerDF)


# %%
corruptDF = spark.createDataFrame([
  (11, 66, 5),
  (12, 68, None),
  (1, None, 6),
  (2, 72, 7)], 
  ["hour", "temperature", "wind"]
)

display(corruptDF)


# %%
corruptDroppedDF = corruptDF.dropna("any")

display(corruptDroppedDF)


# %%
corruptImputedDF = corruptDF.na.fill({"temperature": 68, "wind": 6})

display(corruptImputedDF)


# %%
duplicateDF = spark.createDataFrame([
  (15342, "Conor", "red"),
  (15342, "conor", "red"),
  (12512, "Dorothy", "blue"),
  (5234, "Doug", "aqua")], 
  ["id", "name", "favorite_color"]
)

display(duplicateDF)


# %%
duplicateDedupedDF = duplicateDF.dropDuplicates(["id", "favorite_color"])

display(duplicateDedupedDF)


# %%
# ANSWER
dupedDF = (spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", ":")
    .csv("/mnt/training/dataframes/people-with-dups.txt")
)
display(dupedDF)


# %%
# TEST - Run this cell to test your solution
cols = set(dupedDF.columns)

dbTest("ET2-P-02-01-01", 103000, dupedDF.count())
dbTest("ET2-P-02-01-02", True, "salary" in cols and "lastName" in cols)

print("Tests passed!")


# %%
# ANSWER
from pyspark.sql.functions import col, lower, translate

dupedWithColsDF = (dupedDF
  .select(col("*"),
    lower(col("firstName")).alias("lcFirstName"),
    lower(col("lastName")).alias("lcLastName"),
    lower(col("middleName")).alias("lcMiddleName"),
    translate(col("ssn"), "-", "").alias("ssnNums")
))
display(dupedWithColsDF)


# %%
# ANSWER
dedupedDF = (dupedWithColsDF
  .dropDuplicates(["lcFirstName", "lcMiddleName", "lcLastName", "ssnNums", "gender", "birthDate", "salary"])
  .drop("lcFirstName", "lcMiddleName", "lcLastName", "ssnNums")
)

display(dedupedDF) # should be 100k records


# %%
def manual_split(x):
  return x.split("e")

manual_split("this is my example string")


# %%
from pyspark.sql.types import StringType

manualSplitPythonUDF = spark.udf.register("manualSplitSQLUDF", manual_split, StringType())


# %%
from pyspark.sql.functions import sha1, rand
randomDF = (spark.range(1, 10000 * 10 * 10 * 10)
  .withColumn("random_value", rand(seed=10).cast("string"))
  .withColumn("hash", sha1("random_value"))
  .drop("random_value")
)

display(randomDF)


# %%
randomAugmentedDF = randomDF.select("*", manualSplitPythonUDF("hash").alias("augmented_col"))

display(randomAugmentedDF)


# %%
randomDF.createOrReplaceTempView("randomTable")


# %%
from pyspark.sql.functions import col, rand

randomFloatsDF = (spark.range(0, 100 * 1000 * 1000)
  .withColumn("id", (col("id") / 1000).cast("integer"))
  .withColumn("random_float", rand())
)

randomFloatsDF.cache()
randomFloatsDF.count()

display(randomFloatsDF)


# %%
# ANSWER
from pyspark.sql.types import DoubleType, StructField, StructType

schema = StructType([
  StructField("fahrenheit", DoubleType(), False),
  StructField("celsius", DoubleType(), False),
  StructField("kelvin", DoubleType(), False)
])


# %%
labelsDF = spark.read.parquet("/mnt/training/day-of-week")

display(labelsDF)


# %%
from pyspark.sql.functions import col, date_format

pageviewsDF = (spark.read
  .parquet("/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet/")
  .withColumn("dow", date_format(col("timestamp"), "u").alias("dow"))
)

display(pageviewsDF)


# %%
pageviewsEnhancedDF = pageviewsDF.join(labelsDF, "dow")

display(pageviewsEnhancedDF)


# %%
from pyspark.sql.functions import col

aggregatedDowDF = (pageviewsEnhancedDF
  .groupBy(col("dow"), col("longName"), col("abbreviated"), col("shortName"))  
  .sum("requests")                                             
  .withColumnRenamed("sum(requests)", "Requests")
  .orderBy(col("dow"))
)

display(aggregatedDowDF)


# %%
aggregatedDowDF.explain()


# %%
threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
print("Threshold: {0:,}".format( int(threshold) ))


# %%
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)


# %%
pageviewsDF.join(labelsDF, "dow").explain()


# %%
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", threshold)


# %%
from pyspark.sql.functions import broadcast

pageviewsDF.join(broadcast(labelsDF), "dow").explain()


# %%
# ANSWER
countryLookupDF = spark.read.parquet("/mnt/training/countries/ISOCountryCodes/ISOCountryLookup.parquet")
logWithIPDF = spark.read.parquet("/mnt/training/EDGAR-Log-20170329/enhanced/logDFwithIP.parquet")

display(countryLookupDF)
display(logWithIPDF)


# %%
# ANSWER
from pyspark.sql.functions import broadcast

logWithIPEnhancedDF = (logWithIPDF
  .join(broadcast(countryLookupDF), logWithIPDF.IPLookupISO2 == countryLookupDF.alpha2Code)
  .drop("alpha2Code", "alpha3Code", "numericCode", "ISO31662SubdivisionCode", "independentTerritory")
)

display(logWithIPEnhancedDF)


# %%
wikiDF = (spark.read
  .parquet("/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet")
)
display(wikiDF)


# %%
partitions = wikiDF.rdd.getNumPartitions()
print("Partitions: {0:,}".format( partitions ))


# %%
repartitionedWikiDF = wikiDF.repartition(16)
print("Partitions: {0:,}".format( repartitionedWikiDF.rdd.getNumPartitions() ))


# %%
coalescedWikiDF = repartitionedWikiDF.coalesce(2)
print("Partitions: {0:,}".format( coalescedWikiDF.rdd.getNumPartitions() ))


# %%
spark.conf.get("spark.sql.shuffle.partitions")


# %%
spark.conf.set("spark.sql.shuffle.partitions", "8")


# %%
orderByPartitions = coalescedWikiDF.orderBy("requests").rdd.getNumPartitions()
print("Partitions: {0:,}".format( orderByPartitions ))


# %%
spark.conf.set("spark.sql.shuffle.partitions", "200")


# %%
targetPath = f"{workingDir}/wiki.parquet"

wikiDF.write.mode("OVERWRITE").parquet(targetPath)


# %%
def printRecordsPerPartition(df):
  '''
  Utility method to count & print the number of records in each partition
  '''
  print("Per-Partition Counts:")
  
  def countInPartition(iterator): 
    yield __builtin__.sum(1 for _ in iterator)
    
  results = (df.rdd                   # Convert to an RDD
    .mapPartitions(countInPartition)  # For each partition, count
    .collect()                        # Return the counts to the driver
  )

  for result in results: 
    print("* " + str(result))


# %%
df = spark.range(1, 100)

display(df)


# %%
df.write.mode("OVERWRITE").saveAsTable("myTableManaged")


# %%
get_ipython().run_line_magic('sql', '')
DESCRIBE EXTENDED myTableManaged


# %%
unmanagedPath = f"{workingDir}/myTableUnmanaged"

df.write.mode("OVERWRITE").option('path', unmanagedPath).saveAsTable("myTableUnmanaged")


# %%
hivePath = f"dbfs:/user/hive/warehouse/{databaseName}.db/mytablemanaged"

display(dbutils.fs.ls(hivePath))


# %%



# %%
from pyspark.sql.types import BooleanType, IntegerType, StructType, StringType, StructField, TimestampType

schema = (StructType()
  .add("timestamp", TimestampType())
  .add("url", StringType())
  .add("userURL", StringType())
  .add("pageURL", StringType())
  .add("isNewPage", BooleanType())
  .add("geocoding", StructType()
    .add("countryCode2", StringType())
    .add("city", StringType())
    .add("latitude", StringType())
    .add("country", StringType())
    .add("longitude", StringType())
    .add("stateProvince", StringType())
    .add("countryCode3", StringType())
    .add("user", StringType())
    .add("namespace", StringType()))
)


# %%
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType

kafkaCleanDF = (kafkaDF
  .select(from_json(col("value").cast(StringType()), schema).alias("message"))
  .select("message.*")
)


# %%
# ANSWER

token = "1234"
domain = "https://example.cloud.databricks.com/api/2.0/"

header = {'Authorization': "Bearer "+ token}


# %%
from pyspark.sql.functions import col 

staging_table = (spark.read.parquet("/mnt/training/EDGAR-Log-20170329/enhanced/EDGAR-Log-20170329-sample.parquet/")
  .dropDuplicates(['ip', 'date', 'time']))

production_table = staging_table.sample(.2, seed=123)


# %%
production_table.count() / staging_table.count()


# %%
failedDF = staging_table.join(production_table, on=["ip", "date", "time"], how="left_anti")


# %%
fullDF = production_table.union(failedDF)


# %%
staging_table.count() == fullDF.count()


# %%
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")

display(pagecountsEnAllDF)    


# %%
uncompressedPath = f"{workingDir}/pageCountsUncompressed.csv"
snappyPath =       f"{workingDir}/pageCountsSnappy.csv"
gzipPath =         f"{workingDir}/pageCountsGZIP.csv"

pagecountsEnAllDF.write.mode("OVERWRITE").csv(uncompressedPath)
pagecountsEnAllDF.write.mode("OVERWRITE").option("compression", "snappy").csv(snappyPath)
pagecountsEnAllDF.write.mode("OVERWRITE").option("compression", "GZIP").csv(gzipPath)


# %%
get_ipython().run_line_magic('python', '')
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/") 

display(pagecountsEnAllDF)    


# %%
get_ipython().run_line_magic('python', '')
get_ipython().run_line_magic('timeit', 'pagecountsEnAllDF.count()')


# %%
get_ipython().run_line_magic('python', '')
(pagecountsEnAllDF
  .cache()         # Mark the DataFrame as cached
  .count()         # Materialize the cache
) 


# %%
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")
pagecountsEnAllDF.cache()
pagecountsEnAllDF.count()


# %%
from time import time

def write_read_time(df, file_type, partitions=1, compression=None, outputPath=f"{workingDir}/comparisonTest"):
  '''
  Prints write time and read time for a given DataFrame with given params
  '''
  start_time = time()
  _df = df.repartition(partitions).write.mode("OVERWRITE")
  
  if compression:
    _df = _df.option("compression", compression)
  if file_type == "csv":
    _df.csv(outputPath)
  elif file_type == "parquet":
    _df.parquet(outputPath)
    
  total_time = round(time() - start_time, 1)
  print("Save time of {}s for\tfile_type: {}\tpartitions: {}\tcompression: {}".format(total_time, file_type, partitions, compression))
  
  start_time = time()
  if file_type == "csv":
    spark.read.csv(outputPath).count()
  elif file_type == "parquet":
    spark.read.parquet(outputPath).count()
    
  total_time = round(time() - start_time, 2)
  print("\tRead time of {}s".format(total_time))
  
  
def time_all(df, file_type_list=["csv", "parquet"], partitions_list=[1, 16, 32, 64], compression_list=[None, "gzip", "snappy"]):
  '''
  Wrapper function for write_read_time() to gridsearch lists of file types, partitions, and compression types
  '''
  for file_type in file_type_list:
    for partitions in partitions_list:
      for compression in compression_list:
        write_read_time(df, file_type, partitions, compression)


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Let us check if the sparkcontext is present
# %%
sc


# %%
# Distribute the data - Create a RDD
lines = sc.textFile("/FileStore/tables/shakespeare.txt")

# Create a list with all words, Create tuple (word,1), reduce by key i.e. the word
counts = (lines.flatMap(lambda x: x.split(' '))
          .map(lambda x: (x, 1))
          .reduceByKey(lambda x, y: x + y))

# get the output on local
output = counts.take(10)
# print output
for (word, count) in output:
    print("%s: %i" % (word, count))

# %% [markdown]
# # Functional Programming with Python

# %%
#MAP

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Lets say I want to square each term in my_list.
squared_list = map(lambda x: x**2, my_list)
print(list(squared_list))


# %%
def squared(x):
    return x**2


my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Lets say I want to square each term in my_list.
squared_list = map(squared, my_list)
print(list(squared_list))


# %%
#Filter

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Lets say I want only the even numbers in my list.
filtered_list = filter(lambda x: x % 2 == 0, my_list)
print(list(filtered_list))


# %%
#reduce
my_list = [1, 2, 3, 4, 5]
# Lets say I want to sum all elements in my list.
sum_list = functools.reduce(lambda x, y: x+y, my_list)
print(sum_list)

# %% [markdown]
# # sc.parallelize usage

# %%
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_rdd = sc.parallelize(data, 4)
new_rdd

# %% [markdown]
# # Understanding Transformations
# %% [markdown]
# ### 1.Map

# %%
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(data, 4)
squared_rdd = rdd.map(lambda x: x**2)
squared_rdd.collect()

# %% [markdown]
# ## 2.Filter

# %%
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(data, 4)
filtered_rdd = rdd.filter(lambda x: x % 2 == 0)
filtered_rdd.collect()

# %% [markdown]
# ## 3.distinct

# %%
data = [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8, 9, 10]
rdd = sc.parallelize(data, 4)
distinct_rdd = rdd.distinct()
distinct_rdd.collect()

# %% [markdown]
# ## 4.flatmap

# %%
data = [1, 2, 3, 4]
rdd = sc.parallelize(data, 4)
flat_rdd = rdd.flatMap(lambda x: [x, x**3])
flat_rdd.collect()

# %% [markdown]
# ##5.reducebykey

# %%
data = [('Apple', 'Fruit', 200), ('Banana', 'Fruit', 24), ('Tomato',
                                                           'Fruit', 56), ('Potato', 'Vegetable', 103), ('Carrot', 'Vegetable', 34)]
rdd = sc.parallelize(data, 4)


# %%
category_price_rdd = rdd.map(lambda x: (x[1], x[2]))
category_price_rdd.collect()


# %%
category_total_price_rdd = category_price_rdd.reduceByKey(lambda x, y: x+y)
category_total_price_rdd.collect()

# %% [markdown]
# ## 6. GroupByKey

# %%
data = [('Apple', 'Fruit', 200), ('Banana', 'Fruit', 24), ('Tomato',
                                                           'Fruit', 56), ('Potato', 'Vegetable', 103), ('Carrot', 'Vegetable', 34)]
rdd = sc.parallelize(data, 4)
category_product_rdd = rdd.map(lambda x: (x[1], x[0]))
category_product_rdd.collect()


# %%
grouped_products_by_category_rdd = category_product_rdd.groupByKey()
findata = grouped_products_by_category_rdd.collect()
for data in findata:
    print(data[0], list(data[1]))

# %% [markdown]
# # Actions
# %% [markdown]
# ## 1. reduce

# %%
rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.reduce(lambda x, y: x+y)

# %% [markdown]
# ## 2.take

# %%
rdd = sc.parallelize([1, 2, 3, 4, 5])
rdd.take(3)

# %% [markdown]
# ## 3.takeOrdered

# %%
rdd = sc.parallelize([5, 3, 12, 23])
# descending order
rdd.takeOrdered(3, lambda s: -1*s)


# %%
rdd = sc.parallelize([(5, 23), (3, 34), (12, 344), (23, 29)])
# descending order
rdd.takeOrdered(3, lambda s: -1*s[1])

# %% [markdown]
# # Spark in Action

# %%
userRDD = sc.textFile("/FileStore/tables/u.user")
ratingRDD = sc.textFile("/FileStore/tables/u.data")
movieRDD = sc.textFile("/FileStore/tables/u.item")
print("userRDD:", userRDD.take(1))
print("ratingRDD:", ratingRDD.take(1))
print("movieRDD:", movieRDD.take(1))

# %% [markdown]
# ## 25 most rated movie titles

# %%
# Create a RDD from RatingRDD that only contains the two columns of interest i.e. movie_id,rating.
RDD_movid_rating = ratingRDD.map(
    lambda x: (x.split("\t")[1], x.split("\t")[2]))
print("RDD_movid_rating:", RDD_movid_rating.take(4))

# Create a RDD from MovieRDD that only contains the two columns of interest i.e. movie_id,title.
RDD_movid_title = movieRDD.map(lambda x: (x.split("|")[0], x.split("|")[1]))
print("RDD_movid_title:", RDD_movid_title.take(2))

# merge these two pair RDDs based on movie_id. For this we will use the transformation leftOuterJoin()
rdd_movid_title_rating = RDD_movid_rating.leftOuterJoin(RDD_movid_title)
print("rdd_movid_title_rating:", rdd_movid_title_rating.take(1))

# use the RDD in previous step to create (movie,1) tuple pair RDD
rdd_title_rating = rdd_movid_title_rating.map(lambda x: (x[1][1], 1))
print("rdd_title_rating:", rdd_title_rating.take(2))

# Use the reduceByKey transformation to reduce on the basis of movie_title
rdd_title_ratingcnt = rdd_title_rating.reduceByKey(lambda x, y: x+y)
print("rdd_title_ratingcnt:", rdd_title_ratingcnt.take(2))

# Get the final answer by using takeOrdered Transformation
print("#####################################")
print("25 most rated movies:",
      rdd_title_ratingcnt.takeOrdered(25, lambda x: -x[1]))
print("#####################################")

# %% [markdown]
# We could have done all this in a single command

# %%
print(((ratingRDD.map(lambda x: (x.split("\t")[1], x.split("\t")[2]))).
       leftOuterJoin(movieRDD.map(lambda x: (x.split("|")[0], x.split("|")[1])))).
      map(lambda x: (x[1][1], 1)).
      reduceByKey(lambda x, y: x+y).
      takeOrdered(25, lambda x: -x[1]))

# %% [markdown]
# Find the most highly rated 25 movies using the same dataset. We actually want only those movies which have been rated at least 100 times

# %%
# We already have the RDD rdd_movid_title_rating: [(u'429', (u'5', u'Day the Earth Stood Still, The (1951)'))]
# We create an RDD that contains sum of all the ratings for a particular movie

rdd_title_ratingsum = (rdd_movid_title_rating.
                       map(lambda x: (x[1][1], int(x[1][0]))).
                       reduceByKey(lambda x, y: x+y))

print("rdd_title_ratingsum:", rdd_title_ratingsum.take(2))

# Merge this data with the RDD rdd_title_ratingcnt we created in the last step
# And use Map function to divide ratingsum by rating count.

rdd_title_ratingmean_rating_count = (rdd_title_ratingsum.
                                     leftOuterJoin(rdd_title_ratingcnt).
                                     map(lambda x: (x[0], (float(x[1][0])/x[1][1], x[1][1]))))

print("rdd_title_ratingmean_rating_count:",
      rdd_title_ratingmean_rating_count.take(1))

# We could use take ordered here only but we want to only get the movies which have count
# of ratings more than or equal to 100 so lets filter the data RDD.
rdd_title_rating_rating_count_gt_100 = (rdd_title_ratingmean_rating_count.
                                        filter(lambda x: x[1][1] >= 100))

print("rdd_title_rating_rating_count_gt_100:",
      rdd_title_rating_rating_count_gt_100.take(1))

# Get the final answer by using takeOrdered Transformation
print("#####################################")
print("25 highly rated movies:")
print(rdd_title_rating_rating_count_gt_100.takeOrdered(25, lambda x: -x[1][0]))
print("#####################################")

# %% [markdown]
# #Spark DataFrames
# %% [markdown]
# ## 1.Reading file

# %%
ratings = spark.read.load("/FileStore/tables/u.data",
                          format="csv", sep="\t", inferSchema="true", header="false")

# %% [markdown]
# ## 2. Show File

# %%
ratings.show()


# %%
display(ratings)

# %% [markdown]
# ## 3. Change Column names

# %%
ratings = ratings.toDF(*['user_id', 'movie_id', 'rating', 'unix_timestamp'])


# %%
display(ratings)

# %% [markdown]
# ## 4. Basic Stats

# %%
print(ratings.count())  # // Row Count
print(len(ratings.columns))  # //Column Count


# %%
display(ratings.describe())

# %% [markdown]
# ##5. Select few columns

# %%
display(ratings.select('user_id', 'movie_id'))

# %% [markdown]
# ##6. Filter

# %%
display(ratings.filter(ratings.rating == 5))


# %%
display(ratings.filter((ratings.rating == 5) & (ratings.user_id == 253)))

# %% [markdown]
# ## 7. Groupby

# %%
display(ratings.groupBy("user_id").agg(F.count("user_id"), F.mean("rating")))

# %% [markdown]
# ## 8. Sort

# %%
display(ratings.sort("user_id"))


# %%
#descending Sort
display(ratings.sort(F.desc("user_id")))

# %% [markdown]
# ## JOINS and Merging with Spark Dataframes

# %%
# Let us try to run some SQL on Ratings
ratings.registerTempTable('ratings_table')
newDF = sqlContext.sql('select * from ratings_table where rating>4')
display(newDF)


# %%
#get one more dataframe to join
movies = spark.read.load("/FileStore/tables/u.item",
                         format="csv", sep="|", inferSchema="true", header="false")
movies = movies.toDF(*["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation ", "Children",
                       "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", "War", "Western"])


# %%
display(movies)


# %%
#Let us try joins
movies.registerTempTable('movies_table')
display(sqlContext.sql('select ratings_table.*,movies_table.movie_title from ratings_table left join movies_table on movies_table.movie_id = ratings_table.movie_id'))


# %%
# top 25 most rated movies:


# %%
mostrateddf = sqlContext.sql('select movie_id,movie_title, count(user_id) as num_ratings from (select ratings_table.*,movies_table.movie_title from ratings_table left join movies_table on movies_table.movie_id = ratings_table.movie_id)A group by movie_id,movie_title order by num_ratings desc ')

display(mostrateddf)


# %%
# top 25 highest rated movies having more than 100 votes:

highrateddf = sqlContext.sql('select movie_id,movie_title, avg(rating) as avg_rating,count(movie_id) as num_ratings from (select ratings_table.*,movies_table.movie_title from ratings_table left join movies_table on movies_table.movie_id = ratings_table.movie_id)A group by movie_id,movie_title having num_ratings>100 order by avg_rating desc ')

display(highrateddf)


# %%
display(highrateddf)

# %% [markdown]
# # Converting back and forth from RDD to DF

# %%
highratedrdd = highrateddf.rdd
highratedrdd.take(2)


# %%
data = [('A', 1), ('B', 2), ('C', 3), ('D', 4)]
rdd = sc.parallelize(data)
rdd_new = rdd.map(lambda x: Row(key=x[0], value=int(x[1])))
rdd_as_df = sqlContext.createDataFrame(rdd_new)
display(rdd_as_df)


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %% [markdown]
# # 1. Simple Functions
# %% [markdown]
# ## Read Data

# %%
cases = spark.read.load("/home/rahul/projects/sparkdf/coronavirusdataset/Case.csv",
                        format="csv", sep=",", inferSchema="true", header="true")

# %% [markdown]
# ## Show Data

# %%
cases.show()


# %%
cases.limit(10).toPandas()

# %% [markdown]
# ## Change Column Names

# %%
cases = cases.withColumnRenamed("infection_case", "infection_source")


# %%
cases = cases.toDF(*['case_id', 'province', 'city', 'group', 'infection_case', 'confirmed',
                     'latitude', 'longitude'])

# %% [markdown]
# ## Select Columns

# %%
cases = cases.select('province', 'city', 'infection_case', 'confirmed')


# %%
cases.show()

# %% [markdown]
# ## Sort

# %%
cases.sort("confirmed").show()


# %%
# descending Sort
cases.sort(F.desc("confirmed")).show()

# %% [markdown]
# ## Cast

# %%


# %%
cases = cases.withColumn('confirmed', F.col('confirmed').cast(IntegerType()))
cases = cases.withColumn('city', F.col('city').cast(StringType()))

# %% [markdown]
# ## Filter

# %%
cases.filter((cases.confirmed > 10) & (cases.province == 'Daegu')).show()

# %% [markdown]
# ## GroupBy

# %%

cases.groupBy(["province", "city"]).agg(
    F.sum("confirmed"), F.max("confirmed")).show()


# %%
cases.groupBy(["province", "city"]).agg(F.sum("confirmed").alias(
    "TotalConfirmed"), F.max("confirmed").alias("MaxFromOneConfirmedCase")).show()


# %%
cases.groupBy(["province", "city"]).agg(
    F.sum("confirmed").alias("TotalConfirmed"),
    F.max("confirmed").alias("MaxFromOneConfirmedCase")
).show()

# %% [markdown]
# ## Joins

# %%
regions = spark.read.load("/home/rahul/projects/sparkdf/coronavirusdataset/Region.csv",
                          format="csv", sep=",", inferSchema="true", header="true")
regions.limit(10).toPandas()


# %%
cases = cases.join(regions, ['province', 'city'], how='left')


# %%
cases.limit(10).toPandas()

# %% [markdown]
# # 2. Broadcast/Map Side Joins

# %%
big = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 2, 2],
                    'price': [i for i in range(8)]})

small = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2], 'agg': [
                     'sum', 'mean', 'max', 'min', 'sum', 'mean']})


# %%
big


# %%
small


# %%
big.merge(small, on=['A'], how='left')

# %% [markdown]
# Such sort of operations is aplenty in Spark where you might want to apply multiple operations to a particular key. But assuming that the key data in the Big table is large, it will involve a lot of data movement. And sometimes so much that the application itself breaks. A small optimization then you can do when joining on such big tables(assuming the other table is small) is to broadcast the small table to each machine when you perform a join. You can do this easily using the broadcast keyword.

# %%
cases = cases.join(broadcast(regions), ['province', 'city'], how='left')


# %%
cases

# %% [markdown]
# # 3. Using SQL with Spark

# %%
# Reading Original Cases Back again
cases = spark.read.load("/home/rahul/projects/sparkdf/coronavirusdataset/Case.csv",
                        format="csv", sep=",", inferSchema="true", header="true")


# %%
cases.registerTempTable('cases_table')
newDF = sqlContext.sql('select * from cases_table where confirmed>100')


# %%
newDF.show()

# %% [markdown]
# # 4. Create New Columns
# %% [markdown]
# ## Using Spark Native Functions

# %%
casesWithNewConfirmed = cases.withColumn(
    "NewConfirmed", 100 + F.col("confirmed"))
casesWithNewConfirmed.show()


# %%
casesWithExpConfirmed = cases.withColumn("ExpConfirmed", F.exp("confirmed"))
casesWithExpConfirmed.show()

# %% [markdown]
# ## Spark UDFs

# %%


def casesHighLow(confirmed):
    if confirmed < 50:
        return 'low'
    else:
        return 'high'


#convert to a UDF Function by passing in the function and return type of function
casesHighLowUDF = F.udf(casesHighLow, StringType())

CasesWithHighLow = cases.withColumn("HighLow", casesHighLowUDF("confirmed"))
CasesWithHighLow.show()

# %% [markdown]
# ## Using RDDs

# %%


def rowwise_function(row):
    # convert row to python dictionary:
    row_dict = row.asDict()
    # Add a new key in the dictionary with the new column name and value.
    # This might be a big complex function.
    row_dict['expConfirmed'] = float(np.exp(row_dict['confirmed']))
    # convert dict to row back again:
    newrow = Row(**row_dict)
    # return new row
    return newrow


# convert cases dataframe to RDD
cases_rdd = cases.rdd

# apply our function to RDD
cases_rdd_new = cases_rdd.map(lambda row: rowwise_function(row))

# Convert RDD Back to DataFrame
casesNewDf = sqlContext.createDataFrame(cases_rdd_new)

casesNewDf.show()

# %% [markdown]
# ## Pandas UDF

# %%
cases.printSchema()


# %%

# Declare the schema for the output of our function

outSchema = StructType([StructField('case_id', IntegerType(), True),
                        StructField('province', StringType(), True),
                        StructField('city', StringType(), True),
                        StructField('group', BooleanType(), True),
                        StructField('infection_case', StringType(), True),
                        StructField('confirmed', IntegerType(), True),
                        StructField('latitude', StringType(), True),
                        StructField('longitude', StringType(), True),
                        StructField('normalized_confirmed', DoubleType(), True)
                        ])
# decorate our function with pandas_udf decorator
@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.confirmed
    v = v - v.mean()
    pdf['normalized_confirmed'] = v
    return pdf


confirmed_groupwise_normalization = cases.groupby(
    "infection_case").apply(subtract_mean)

confirmed_groupwise_normalization.limit(10).toPandas()

# %% [markdown]
# # 5. Spark Window Functions

# %%
timeprovince = spark.read.load("/home/rahul/projects/sparkdf/coronavirusdataset/TimeProvince.csv",
                               format="csv",                         sep=",", inferSchema="true", header="true")
timeprovince.show()

# %% [markdown]
# # Ranking

# %%
windowSpec = Window().partitionBy(['province']).orderBy(F.desc('confirmed'))
cases.withColumn("rank", F.rank().over(windowSpec)).show()

# %% [markdown]
# # Lag

# %%
windowSpec = Window().partitionBy(['province']).orderBy('date')
timeprovinceWithLag = timeprovince.withColumn(
    "lag_7", F.lag("confirmed", 7).over(windowSpec))

timeprovinceWithLag.filter(timeprovinceWithLag.date > '2020-03-10').show()

# %% [markdown]
# # Rolling Aggregations

# %%

windowSpec = Window().partitionBy(
    ['province']).orderBy('date').rowsBetween(-6, 0)
timeprovinceWithRoll = timeprovince.withColumn(
    "roll_7_confirmed", F.mean("confirmed").over(windowSpec))
timeprovinceWithRoll.filter(timeprovinceWithLag.date > '2020-03-10').show()

# %% [markdown]
# ## Running Totals

# %%

windowSpec = Window().partitionBy(['province']).orderBy(
    'date').rowsBetween(Window.unboundedPreceding, Window.currentRow)
timeprovinceWithRoll = timeprovince.withColumn(
    "cumulative_confirmed", F.sum("confirmed").over(windowSpec))
timeprovinceWithRoll.filter(timeprovinceWithLag.date > '2020-03-10').show()

# %% [markdown]
# # 6. Pivot Dataframes

# %%
pivotedTimeprovince = timeprovince.groupBy('date').pivot('province')                       .agg(
    F.sum('confirmed').alias('confirmed'), F.sum('released').alias('released'))
pivotedTimeprovince.limit(10).toPandas()

# %% [markdown]
# # 7. Unpivot/Stack Dataframes

# %%
pivotedTimeprovince.columns


# %%
newColnames = [x.replace("-", "_") for x in pivotedTimeprovince.columns]


# %%
pivotedTimeprovince = pivotedTimeprovince.toDF(*newColnames)


# %%
expression = ""
cnt = 0
for column in pivotedTimeprovince.columns:
    if column != 'date':
        cnt += 1
        expression += f"'{column}' , {column},"

expression = f"stack({cnt}, {expression[:-1]}) as (Type,Value)"


# %%
unpivotedTimeprovince = pivotedTimeprovince.select('date', F.expr(expression))
unpivotedTimeprovince.show()

# %% [markdown]
# # 8. Salting

# %%
cases = cases.withColumn("salt_key", F.concat(
    F.col("infection_case"), F.lit("_"), F.monotonically_increasing_id() % 10))


# %%
cases.show()


# %%
cases_temp = cases.groupBy(["infection_case", "salt_key"]).agg(
    F.sum("confirmed").alias("salt_confirmed"))
cases_temp.show()


# %%
cases_answer = cases_temp.groupBy(["infection_case"]).agg(F.sum("salt_confirmed").alias("final_confirmed"))
cases_answer.show()



# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd


# %%
get_ipython().run_line_magic('run', '"./Includes/Classroom-Setup"')


# %%
path = "/mnt/training/EDGAR-Log-20170329/EDGAR-Log-20170329.csv"

logDF = (spark
  .read
  .option("header", True)
  .csv(path)
  .sample(withReplacement=False, fraction=0.3, seed=3) # using a sample to reduce data size
)

display(logDF)


# %%
from pyspark.sql.functions import col

serverErrorDF = (logDF
  .filter((col("code") >= 500) & (col("code") < 600))
  .select("date", "time", "extention", "code")
)

display(serverErrorDF)


# %%
from pyspark.sql.functions import from_utc_timestamp, hour, col

countsDF = (serverErrorDF
  .select(hour(from_utc_timestamp(col("time"), "GMT")).alias("hour"))
  .groupBy("hour")
  .count()
  .orderBy("hour")
)

display(countsDF)


# %%
workingDir = '.'


# %%
targetPath = workingDir + "/serverErrorDF.parquet"

(serverErrorDF
  .write
  .mode("overwrite") # overwrites a file if it already exists
  .parquet(targetPath)
)


# %%



# %%
# TODO

wikiDf = (spark.read
  .option("delimiter", "\t")
  .option("header", True)
  .option("timestampFormat", "mm/dd/yyyy hh:mm:ss a")
  .option("inferSchema", True)
  .csv("/mnt/training/wikipedia/pageviews/pageviews_by_second.tsv")
)
display(wikiDf)


# %%
get_ipython().run_line_magic('scala', '')
(/, run, this, regardless, of, language, type)
Class.forName("org.postgresql.Driver")


# %%
jdbcHostname = "server1.databricks.training"
jdbcPort = 5432
jdbcDatabase = "training"

jdbcUrl = f"jdbc:postgresql://{jdbcHostname}:{jdbcPort}/{jdbcDatabase}"


# %%
connectionProps = {
  "user": "readonly",
  "password": "readonly"
}


# %%
tableName = "training.people_1m"

peopleDF = spark.read.jdbc(url=jdbcUrl, table=tableName, properties=connectionProps)

display(peopleDF)


# %%
integerDF = spark.range(1000, 10000)

display(integerDF)


# %%
from pyspark.sql.functions import col, max, min

colMin = integerDF.select(min("id")).first()[0]
colMax = integerDF.select(max("id")).first()[0]

normalizedIntegerDF = (integerDF
  .withColumn("normalizedValue", (col("id") - colMin) / (colMax - colMin) )
)

display(normalizedIntegerDF)


# %%
corruptDF = spark.createDataFrame([
  (11, 66, 5),
  (12, 68, None),
  (1, None, 6),
  (2, 72, 7)], 
  ["hour", "temperature", "wind"]
)

display(corruptDF)


# %%
corruptDroppedDF = corruptDF.dropna("any")

display(corruptDroppedDF)


# %%
corruptImputedDF = corruptDF.na.fill({"temperature": 68, "wind": 6})

display(corruptImputedDF)


# %%
duplicateDF = spark.createDataFrame([
  (15342, "Conor", "red"),
  (15342, "conor", "red"),
  (12512, "Dorothy", "blue"),
  (5234, "Doug", "aqua")], 
  ["id", "name", "favorite_color"]
)

display(duplicateDF)


# %%
duplicateDedupedDF = duplicateDF.dropDuplicates(["id", "favorite_color"])

display(duplicateDedupedDF)


# %%
# ANSWER
dupedDF = (spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", ":")
    .csv("/mnt/training/dataframes/people-with-dups.txt")
)
display(dupedDF)


# %%
# TEST - Run this cell to test your solution
cols = set(dupedDF.columns)

dbTest("ET2-P-02-01-01", 103000, dupedDF.count())
dbTest("ET2-P-02-01-02", True, "salary" in cols and "lastName" in cols)

print("Tests passed!")


# %%
# ANSWER
from pyspark.sql.functions import col, lower, translate

dupedWithColsDF = (dupedDF
  .select(col("*"),
    lower(col("firstName")).alias("lcFirstName"),
    lower(col("lastName")).alias("lcLastName"),
    lower(col("middleName")).alias("lcMiddleName"),
    translate(col("ssn"), "-", "").alias("ssnNums")
))
display(dupedWithColsDF)


# %%
# ANSWER
dedupedDF = (dupedWithColsDF
  .dropDuplicates(["lcFirstName", "lcMiddleName", "lcLastName", "ssnNums", "gender", "birthDate", "salary"])
  .drop("lcFirstName", "lcMiddleName", "lcLastName", "ssnNums")
)

display(dedupedDF) # should be 100k records


# %%
def manual_split(x):
  return x.split("e")

manual_split("this is my example string")


# %%
from pyspark.sql.types import StringType

manualSplitPythonUDF = spark.udf.register("manualSplitSQLUDF", manual_split, StringType())


# %%
from pyspark.sql.functions import sha1, rand
randomDF = (spark.range(1, 10000 * 10 * 10 * 10)
  .withColumn("random_value", rand(seed=10).cast("string"))
  .withColumn("hash", sha1("random_value"))
  .drop("random_value")
)

display(randomDF)


# %%
randomAugmentedDF = randomDF.select("*", manualSplitPythonUDF("hash").alias("augmented_col"))

display(randomAugmentedDF)


# %%
randomDF.createOrReplaceTempView("randomTable")


# %%
from pyspark.sql.functions import col, rand

randomFloatsDF = (spark.range(0, 100 * 1000 * 1000)
  .withColumn("id", (col("id") / 1000).cast("integer"))
  .withColumn("random_float", rand())
)

randomFloatsDF.cache()
randomFloatsDF.count()

display(randomFloatsDF)


# %%
# ANSWER
from pyspark.sql.types import DoubleType, StructField, StructType

schema = StructType([
  StructField("fahrenheit", DoubleType(), False),
  StructField("celsius", DoubleType(), False),
  StructField("kelvin", DoubleType(), False)
])


# %%
labelsDF = spark.read.parquet("/mnt/training/day-of-week")

display(labelsDF)


# %%
from pyspark.sql.functions import col, date_format

pageviewsDF = (spark.read
  .parquet("/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet/")
  .withColumn("dow", date_format(col("timestamp"), "u").alias("dow"))
)

display(pageviewsDF)


# %%
pageviewsEnhancedDF = pageviewsDF.join(labelsDF, "dow")

display(pageviewsEnhancedDF)


# %%
from pyspark.sql.functions import col

aggregatedDowDF = (pageviewsEnhancedDF
  .groupBy(col("dow"), col("longName"), col("abbreviated"), col("shortName"))  
  .sum("requests")                                             
  .withColumnRenamed("sum(requests)", "Requests")
  .orderBy(col("dow"))
)

display(aggregatedDowDF)


# %%
aggregatedDowDF.explain()


# %%
threshold = spark.conf.get("spark.sql.autoBroadcastJoinThreshold")
print("Threshold: {0:,}".format( int(threshold) ))


# %%
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)


# %%
pageviewsDF.join(labelsDF, "dow").explain()


# %%
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", threshold)


# %%
from pyspark.sql.functions import broadcast

pageviewsDF.join(broadcast(labelsDF), "dow").explain()


# %%
# ANSWER
countryLookupDF = spark.read.parquet("/mnt/training/countries/ISOCountryCodes/ISOCountryLookup.parquet")
logWithIPDF = spark.read.parquet("/mnt/training/EDGAR-Log-20170329/enhanced/logDFwithIP.parquet")

display(countryLookupDF)
display(logWithIPDF)


# %%
# ANSWER
from pyspark.sql.functions import broadcast

logWithIPEnhancedDF = (logWithIPDF
  .join(broadcast(countryLookupDF), logWithIPDF.IPLookupISO2 == countryLookupDF.alpha2Code)
  .drop("alpha2Code", "alpha3Code", "numericCode", "ISO31662SubdivisionCode", "independentTerritory")
)

display(logWithIPEnhancedDF)


# %%
wikiDF = (spark.read
  .parquet("/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet")
)
display(wikiDF)


# %%
partitions = wikiDF.rdd.getNumPartitions()
print("Partitions: {0:,}".format( partitions ))


# %%
repartitionedWikiDF = wikiDF.repartition(16)
print("Partitions: {0:,}".format( repartitionedWikiDF.rdd.getNumPartitions() ))


# %%
coalescedWikiDF = repartitionedWikiDF.coalesce(2)
print("Partitions: {0:,}".format( coalescedWikiDF.rdd.getNumPartitions() ))


# %%
spark.conf.get("spark.sql.shuffle.partitions")


# %%
spark.conf.set("spark.sql.shuffle.partitions", "8")


# %%
orderByPartitions = coalescedWikiDF.orderBy("requests").rdd.getNumPartitions()
print("Partitions: {0:,}".format( orderByPartitions ))


# %%
spark.conf.set("spark.sql.shuffle.partitions", "200")


# %%
targetPath = f"{workingDir}/wiki.parquet"

wikiDF.write.mode("OVERWRITE").parquet(targetPath)


# %%
def printRecordsPerPartition(df):
  '''
  Utility method to count & print the number of records in each partition
  '''
  print("Per-Partition Counts:")
  
  def countInPartition(iterator): 
    yield __builtin__.sum(1 for _ in iterator)
    
  results = (df.rdd                   # Convert to an RDD
    .mapPartitions(countInPartition)  # For each partition, count
    .collect()                        # Return the counts to the driver
  )

  for result in results: 
    print("* " + str(result))


# %%
df = spark.range(1, 100)

display(df)


# %%
df.write.mode("OVERWRITE").saveAsTable("myTableManaged")


# %%
get_ipython().run_line_magic('sql', '')
DESCRIBE EXTENDED myTableManaged


# %%
unmanagedPath = f"{workingDir}/myTableUnmanaged"

df.write.mode("OVERWRITE").option('path', unmanagedPath).saveAsTable("myTableUnmanaged")


# %%
hivePath = f"dbfs:/user/hive/warehouse/{databaseName}.db/mytablemanaged"

display(dbutils.fs.ls(hivePath))


# %%



# %%
from pyspark.sql.types import BooleanType, IntegerType, StructType, StringType, StructField, TimestampType

schema = (StructType()
  .add("timestamp", TimestampType())
  .add("url", StringType())
  .add("userURL", StringType())
  .add("pageURL", StringType())
  .add("isNewPage", BooleanType())
  .add("geocoding", StructType()
    .add("countryCode2", StringType())
    .add("city", StringType())
    .add("latitude", StringType())
    .add("country", StringType())
    .add("longitude", StringType())
    .add("stateProvince", StringType())
    .add("countryCode3", StringType())
    .add("user", StringType())
    .add("namespace", StringType()))
)


# %%
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StringType

kafkaCleanDF = (kafkaDF
  .select(from_json(col("value").cast(StringType()), schema).alias("message"))
  .select("message.*")
)


# %%
# ANSWER

token = "1234"
domain = "https://example.cloud.databricks.com/api/2.0/"

header = {'Authorization': "Bearer "+ token}


# %%
from pyspark.sql.functions import col 

staging_table = (spark.read.parquet("/mnt/training/EDGAR-Log-20170329/enhanced/EDGAR-Log-20170329-sample.parquet/")
  .dropDuplicates(['ip', 'date', 'time']))

production_table = staging_table.sample(.2, seed=123)


# %%
production_table.count() / staging_table.count()


# %%
failedDF = staging_table.join(production_table, on=["ip", "date", "time"], how="left_anti")


# %%
fullDF = production_table.union(failedDF)


# %%
staging_table.count() == fullDF.count()


# %%
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")

display(pagecountsEnAllDF)    


# %%
uncompressedPath = f"{workingDir}/pageCountsUncompressed.csv"
snappyPath =       f"{workingDir}/pageCountsSnappy.csv"
gzipPath =         f"{workingDir}/pageCountsGZIP.csv"

pagecountsEnAllDF.write.mode("OVERWRITE").csv(uncompressedPath)
pagecountsEnAllDF.write.mode("OVERWRITE").option("compression", "snappy").csv(snappyPath)
pagecountsEnAllDF.write.mode("OVERWRITE").option("compression", "GZIP").csv(gzipPath)


# %%
get_ipython().run_line_magic('python', '')
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/") 

display(pagecountsEnAllDF)    


# %%
get_ipython().run_line_magic('python', '')
get_ipython().run_line_magic('timeit', 'pagecountsEnAllDF.count()')


# %%
get_ipython().run_line_magic('python', '')
(pagecountsEnAllDF
  .cache()         # Mark the DataFrame as cached
  .count()         # Materialize the cache
) 


# %%
pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")
pagecountsEnAllDF.cache()
pagecountsEnAllDF.count()


# %%
from time import time

def write_read_time(df, file_type, partitions=1, compression=None, outputPath=f"{workingDir}/comparisonTest"):
  '''
  Prints write time and read time for a given DataFrame with given params
  '''
  start_time = time()
  _df = df.repartition(partitions).write.mode("OVERWRITE")
  
  if compression:
    _df = _df.option("compression", compression)
  if file_type == "csv":
    _df.csv(outputPath)
  elif file_type == "parquet":
    _df.parquet(outputPath)
    
  total_time = round(time() - start_time, 1)
  print("Save time of {}s for\tfile_type: {}\tpartitions: {}\tcompression: {}".format(total_time, file_type, partitions, compression))
  
  start_time = time()
  if file_type == "csv":
    spark.read.csv(outputPath).count()
  elif file_type == "parquet":
    spark.read.parquet(outputPath).count()
    
  total_time = round(time() - start_time, 2)
  print("\tRead time of {}s".format(total_time))
  
  
def time_all(df, file_type_list=["csv", "parquet"], partitions_list=[1, 16, 32, 64], compression_list=[None, "gzip", "snappy"]):
  '''
  Wrapper function for write_read_time() to gridsearch lists of file types, partitions, and compression types
  '''
  for file_type in file_type_list:
    for partitions in partitions_list:
      for compression in compression_list:
        write_read_time(df, file_type, partitions, compression)


