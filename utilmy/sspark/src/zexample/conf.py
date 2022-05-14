conf =  pyspark.SparkConf()
conf.set("spark.sql.tungsten.enabled", "false")
sc = getOrCreateSparkContext(conf)