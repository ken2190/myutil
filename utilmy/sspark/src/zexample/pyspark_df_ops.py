from pyspark.sql import functions as F

# remap one columns value F.when, otherwise
df = df.withColumn('colx', F.when(F.col('prefix')=='EZS', 'EZY').otherwise(F.col('prefix'))) 

# remap one column with double when condition. f.i. Reassign pier_in 'D' to DN/DSN based on incoming gate number
luggage_df = (luggage_df.withColumn('pier_in', 
                 F.when((F.col('vop_in').isin(DS))  & (F.col('pier_in') == 'D'), 'DS')
                  .when((F.col('vop_in').isin(DNS)) & (F.col('pier_in') == 'D'), 'DNS')
                  .otherwise(F.col('pier_in')))

# get unique values from column based on conditions
[x.col_1 for x in df.select(['col_1', 'col_2', 'col_3'])
 .where((F.col('col_2') == 'D') & (F.col('col_3') == 'Y')).distinct().collect()]

# fix for double rows with diff time stamps, keep first 
from pyspark.sql import Window as w
window_cols = ['col_1', 'col_2', 'col_3']
datetime_sch_fix_window = w.partitionBy(window_cols).orderBy(F.col('time_stamp_col').asc())

df = df.withColumn('datetime_col', F.min(F.col('time_stamp_col')).over(datetime_sch_fix_window)) 

# map values in a column to a new one using a dictionary (udf version independent solution)
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

def translate(mapping):
    def translate_(col):
        return mapping.get(col)
    return udf(translate_, StringType())

mapping = {
    'A': 'S', 'B': 'S', 'C': 'S', 'DS': 'S', 'DNS': 'S', 
    'E': 'NS', 'F': 'NS', 'G': 'NS', 'H': 'NS'}

df = df.withColumn("new_col", translate(mapping)("col_to_be_mapped"))