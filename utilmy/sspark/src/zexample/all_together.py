from pyspark.sql import SparkSession
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
    number_of_rows_by_date = df.groupby('date').size()
    number_of_rows_by_date.columns = ['date', 'number_of_rows']
    number_of_rows_by_date['user_id'] = df['user_id'].iloc[:1]

    return number_of_rows_by_date


def load_data(spark, s3_location):
    """
    spark:
        Spark session
    s3_location:
        S3 bucket name and object prefix
    """
    
    return (
        spark
        .read
        .options(
            delimiter=',',
            header=True,
            inferSchema=False,
        )
        .csv(s3_location)
    )


with SparkSession.builder.appName('Mage').getOrCreate() as spark:
    # 1. Load data from S3 files
    df = load_data(spark, 's3://feature-sets/users/profiles/v1/*')
    
    # 2. Group data by 'user_id' column
    grouped = df.groupby('user_id')
    
    # 3. Apply function named 'custom_transformation_function';
    # we will define this function later in this article
    df_transformed = grouped.apply(custom_transformation_function)

    # 4. Save new transformed data to S3
    (
        df_transformed.write
        .option('delimiter', ',')
        .option('header', 'True')
        .mode('overwrite')
        .csv('s3://feature-sets/users/profiles/transformed/v1/*')
    )