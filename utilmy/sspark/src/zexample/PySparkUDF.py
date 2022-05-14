from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

# 1.- UDF with f as a lambda
to_upper = udf(lambda s: s.upper() if s else None, StringType())

# 2.- UDF with f as a method
def to_upper(s):
  if s is not None:
    return s.upper()
to_upper = udf(to_upper)

# 3.- function using the @udf annotation
@udf(returnType=StringType())
def to_upper(s):
  if s is not None:
    return s.upper()