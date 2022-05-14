# get the data into PySpark - we reuse the "model_data_df" defined above
sql_context = SQLContext(spark_context)
spark_df = sql_context.createDataFrame(model_data_df)

# train and score models using PySpark
# we add the computed column “score_result” to the Spark data frame:
score_df = spark_df.withColumn(
    "score_result", spark_train_and_test_udf(spark_df["x"], spark_df["y"])
)

# select columns and convert back to a Pandas data frame
pandas_df = score_df.select(
    "score_result", "pickup_loc_id", "dropoff_loc_id"
).toPandas()

# back in Pandas, decode the Python dictionary within "score_result"
load_obj_from_spark_v = np.vectorize(load_obj_from_spark, otypes=[object])
pandas_df["score_result"] = load_obj_from_spark_v(pandas_df["score_result"])
