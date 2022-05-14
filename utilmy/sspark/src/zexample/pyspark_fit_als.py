from pyspark.sql import Row
from pyspark.sql.functions import udf, desc, percent_rank
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from pyspark.ml.recommendation import ALS

def load(sqlContext):
  """
  Given a tsv file with all users purchase timestamps, build a pySpark dataframe for fitting an ALS model.
  The input .tsv is as follows:

  user_id \t item_id \t latest_timestamp \t frequency

  where:
  user_id - unique id per user
  item_id - unique id per item
  latest_timestamp - timestamp of the purchase
  frequency - quantity purchased
  """
    schema = StructType([
      StructField("user_id", IntegerType()), StructField("item_id", StringType()),
      StructField("latest_timestamp", IntegerType()), StructField("frequency", IntegerType()),
    ])
    # Get example data in this repo: https://github.com/mauhcs/hybrid-recommender
    item_hist    = sqlContext.read.csv("data/item_history.tsv",sep="\t", header=True, schema=schema)
    # Split item history in train, test and set validation aside for later
    
    def get_key_value(p):
      # for grouping as key value
      # Drop latest_timestamp from columns and erase the 'I' in front
      # of the item ids for faster evaluation.
      return (int(p[0]),int(p[1].replace("I",""))), float(p[3])
    
    # Group number of purchase by user id and item
    ihist_train = item_hist.orderBy("latest_timestamp")
    frequencyRDD = ihist_train.rdd.map(get_key_value).reduceByKey(lambda x: sum(list(x)) )
    frequencyRDD = frequencyRDD.map(lambda x: Row(user_id=int(x[0][0]),item_id=int(x[0][1]),frequency=float(x[1])) )
    frequencies = sqlContext.createDataFrame(frequencyRDD)
  return ihist_train

# get a sqlContext with this gist: https://gist.github.com/mauhcs/96564ae80ac4f11e7dfde4be5b99551e
frequencies = load(sqlContext)

# Instantiate an ALS model and fit to data
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="item_id", 
          ratingCol="frequency", coldStartStrategy="drop", rank=5)
model = als.fit(frequencies)

# Recommend 10 items to all users:
recommendations = model.recommendForAllUsers(10)
print(recommendations)
