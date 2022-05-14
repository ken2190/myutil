from pyspark.sql.functions import sqrt
from pyspark.sql.functions import hour, year, month, dayofmonth, dayofweek
from pyspark.sql.functions import udf, col

def clean(spark, df):
      df = df.where((df["pickup_longitude"] >= -75) & (df["pickup_longitude"] <= -73)) \
              .where((df["dropoff_longitude"] >= -75) & (df["dropoff_longitude"] <= -73)) \
              .where((df["pickup_latitude"] >= 39) & (df["pickup_latitude"] <= 42)) \
              .where((df["dropoff_latitude"] >= 39) & (df["dropoff_latitude"] <= 42))
      # Remove possible outliers
      df = df.where((df["fare_amount"] > 0 ) & (df["fare_amount"] <= 250))
      # Remove inconsistent values
      df = df.where((df["dropoff_longitude"] != df["pickup_longitude"]))
      df = df.where((df["dropoff_latitude"] != df["pickup_latitude"]))  
      return df   

@udf("int")
def late_night (hour):
    if (hour <= 6) or (hour >= 20):
        return 1
    else:
        return 0

@udf("int")
def night (hour, weekday):
    if ((hour<= 20) and (hour >= 16)) and (weekday < 5):
        return 1
    else:
        return 0
      
spark.udf.register("late_night_udf", late_night)
spark.udf.register("night_udf", night)

def add_time_features(spark, df):
  time_df = df.select('*', hour('key').cast('int').alias('hour'), \
                           year('key').cast('int').alias('year'), \
                           month('key').cast('int').alias('month'), \
                           dayofmonth('key').cast('int').alias('dayofmonth'), \
                           dayofweek('key').cast('int').alias('dayofweek'))
  time2_df = time_df.select('*', late_night('hour').alias('late_night'), night('hour','dayofweek').alias('night'))
  return time2_df


def add_distance_features(spark, df):
  ef_df = df.selectExpr('*', "(pickup_latitude - dropoff_latitude) as latdiff", "(pickup_longitude - dropoff_longitude) as londiff" )
  ef_df2 = ef_df.selectExpr( '*', " (sqrt(( latdiff * latdiff ) + (londiff * londiff) )) as euclidean")
  ef_df3 = ef_df2.selectExpr( '*', " (abs(latdiff) + abs(londiff)) as manhattan")
  ef_df4 = ef_df3.selectExpr( '*', "(pickup_latitude * pickup_longitude) as ploc", "(dropoff_latitude * dropoff_longitude) as dloc")

  return ef_df4

def convert_and_drop_columns(spark, df):
    co_df = df.select('key','hour','year','month','dayofmonth','dayofweek','late_night','passenger_count','night', \
                      col('fare_amount').cast('float'), col('latdiff').cast('float'), col('londiff').cast('float'), col('euclidean').cast('float'), \
                      col('manhattan').cast('float'), col('ploc').cast('float'), col('dloc').cast('float'))
    return co_df
