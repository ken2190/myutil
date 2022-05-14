from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)


"""
StructField arguments:
    First argument: column name
    Second argument: column type
    Third argument: True if this column can have null values
"""
SCHEMA_COMING_SOON = StructType([
    StructField('user_id', IntegerType(), True),
    StructField('name', StringType(), True),
    StructField('number_of_rows', IntegerType(), True),
])


@pandas_udf(
    SCHEMA_COMING_SOON,
    PandasUDFType.GROUPED_MAP,
)
def custom_transformation_function(df):
    pass
