from pyspark.sql.functions import udf
from pyspark.sql.types import *

schema = StructType([
    StructField("foo", FloatType(), False),
    StructField("bar", FloatType(), False)
])
udf(function(), schema)