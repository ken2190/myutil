from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

# create spark session
spark = SparkSession \
    .builder \
    .appName("Wrangling Data") \
    .getOrCreate()

# read data json
path = "path/mydata.json"
data = spark.read.json(path)
# data exploration

# view 5 first rows
data.take(5)

# view columns
data.printSchema()

# view stats
data.describe("column_name").show()

# number of rows
data.count()

# drop duplicates
data.select("column_name").dropDuplicates().sort("column_name").show()

# select specific columns with a condition
data.select(["column_name1", "column_name2", "column_name3", "column_name4"]).where(data.column_name == "CONDITION").collect()

# convert spark dataframe to pandas

pandas_df = spark_df.toPandas()