```python
from pyspark.sql.types import *

def date_range(t1, t2, step=60*60*24):
    """Return a list of equally spaced points between t1 and t2 with stepsize step."""
    return [t1 + step*x for x in range(int((t2-t1)/step)+1)]

date_range_udf = func.udf(date_range, ArrayType(LongType()))

# create time boundaries for each house
df_base = df.groupBy('house')\
            .agg(func.min('readtime').cast('integer').alias('readtime_min'), 
                 func.max('readtime').cast('integer').alias('readtime_max'))
  
# generate timegrid and explode
df_base = df_base.withColumn("readtime", func.explode(date_range_udf("readtime_min", "readtime_max")))\
                 .drop('readtime_min', 'readtime_max')

# left outer join existing read values
df_all_dates = df_base.join(df, ["house", "readtime"], "leftouter")
```