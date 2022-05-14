from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

def some_func(a: int, b: int) -> int:
	return a+b

# pyspark
udf_some_func = F.udf(some_func, FloatType())

# in SQL
# SELECT *, UDF_SOME_FUNC(col_a, col_b) FROM table
spark.udf.register("udf_some_func", some_func, StringType())
