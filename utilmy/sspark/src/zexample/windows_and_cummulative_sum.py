# build an udf function for a dummy variable
event = udf(lambda x: 1 if x == "CONDITION" else 0, IntegerType())

# create the dummy variable with udf
data = data.withColumn("DUMMY_VARIABLE", event("COLUMN_TO_ONEHOT"))

from pyspark.sql import Window
windowval = Window.partitionBy("Id").orderBy(desc("timestamp")).rangeBetween(Window.unboundedPreceding, 0)
data = data.withColumn("phase", Fsum("new_variable").over(windowval))