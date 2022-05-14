data_frame.withColumn(
    "prediction",
    predict_pandas_udf(col("feature1"), col("feature2"), ...)
)