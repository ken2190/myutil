from pyspark.sql.functions import pandas_udf, PandasUDFType


@pandas_udf(
    SCHEMA_COMING_SOON,
    PandasUDFType.GROUPED_MAP,
)
def custom_transformation_function(df):
    pass
