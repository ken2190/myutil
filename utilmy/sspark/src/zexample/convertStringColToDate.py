from pyspark.sql import functions as F
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType


newDF = df.withColumn("Date", F.to_date(col("DATE"), "yyyy-MM-dd"))