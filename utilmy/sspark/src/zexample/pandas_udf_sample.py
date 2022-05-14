from pyspark.sql.functions import pandas_udf

#...


# Use pandas_udf to define a Pandas UDF
@pandas_udf('string')
# Input/output are both a pandas.Series of string
def pandas_not_null(s):
    return s.fillna("_NO_₦Ӑ_").replace('', '_NO_ӖӍΡṬΫ_')