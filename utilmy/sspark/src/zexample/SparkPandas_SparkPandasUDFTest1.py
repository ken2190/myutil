from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
import pandas as pd


sourcedf = spark.read\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .csv("file:///SparkCourse/ml-100k/Meteorite_Landings.csv")


def get_var_category(series):
    unique_count = series.nunique(dropna=False)
    total_count = len(series)
    if pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    elif pd.api.types.is_datetime64_dtype(series):
        return 'Date'
    elif unique_count==total_count:
        return 'Text (Unique)'
    else:
        return 'Text'


def qtd_nulls(series):
    return len(series) - series.count()


def print_categories(df):
  for column_name in df.columns:
      print(column_name, ": ", get_var_category(df[column_name]))
      print("Nulls", ": ", qtd_nulls(df[column_name]))


print_categories(sourcedf)