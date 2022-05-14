# Date manipulations
import datetime as dt

# Create a function that returns string from a timestamp 
def format_timestamp(ts):
    return dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

# Create a date formating UDF
format_timestamp_udf = udf(lambda x: format_timestamp(x/1000.0))

# new date column outputs
vector_matrix_df = vector_matrix_df.withColumn("str_date", format_timestamp_udf(vector_matrix_df.ts))
vector_matrix_df = vector_matrix_df.withColumn("date", to_date(vector_matrix_df.str_date))
vector_matrix_df=(vector_matrix_df.withColumn('dayssinceJan11900',F.datediff(vector_matrix_df.date,F.lit(dt.datetime(1900, 1, 1)))))