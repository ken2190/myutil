## get normalized bonus grouped by department
from pyspark.sql import types as T

# Declare the schema for the output of our function
outSchema = T.StructType([T.StructField('employee_name', T.StringType(),True),
                        T.StructField('age', T.LongType(),True),
                        T.StructField('state', T.StringType(),True),
                        T.StructField('salary', T.LongType(),True),
                        T.StructField('salary_in_k', T.DoubleType(),True),
                        T.StructField('department', T.StringType(),True),
                        T.StructField('bonus', T.LongType(),True),
                        T.StructField('normalized_bonus', T.DoubleType(),True)
                       ])

# decorate our function with pandas_udf decorator
@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.bonus
    v = v - v.mean()
    pdf['normalized_bonus'] = v
    
    return pdf

confirmed_groupwise_normalization = df.groupby("department").apply(subtract_mean)

confirmed_groupwise_normalization.toPandas()