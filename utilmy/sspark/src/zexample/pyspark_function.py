from pyspark.sql.types import StructType

def max_time(df):
    df['Time'] = pd.to_datetime(df['Time'])
    return str(df.Time.max())

def apply_to_group(group_col, fxn):
    
#     @spark_udf(IntegerType())
#     def _apply_fxn(struct_col):
#         first_row = struct_col[0]
#         col_names = list(first_row.asDict().keys())
#         df = pd.DataFrame(struct_col, columns = col_names)
#         return fxn(df)
    
    @spark_udf(StringType())
    def _apply_fxn(struct_col):
        first_row = struct_col[0]
        col_names = first_row.__fields__
        df = pd.DataFrame(struct_col, columns = col_names)
        return fxn(df)
    return _apply_fxn(group_col)
