from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql import Window as w
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np
import datetime

sequence = [1, 2, 4, 7, 11, 16, 22]
sequence2 = [3, 5, 6, 8, 10, 18, 20]

# create two example time series with different ticks and inconsistent spacing
df = spark.createDataFrame(
  list(zip(
    ['series 1'] * len(sequence),
    [datetime.datetime(2021,1,1,0,i) for i in sequence],
    [i**2*np.random.uniform(0,1) for i in sequence]
  ))
  + list(zip(
    ['series 2'] * len(sequence2),
    [datetime.datetime(2021,1,1,0,i) for i in sequence2],
    [i**2*np.random.uniform(0,1) for i in sequence2]
  )),
  schema=t.StructType([
    t.StructField('id', t.StringType()),
    t.StructField('time', t.TimestampType()),
    t.StructField('value', t.FloatType())
  ])
)

# up sample the time series to regular tickspacing
regular_ticks = (
  df.groupBy('id')
  .agg(
    f.min('time').alias('min_time'),
    f.max('time').alias('max_time'),
  )
  .withColumn('time', f.explode(f.sequence('min_time', 'max_time', f.lit('1 minute').cast('Interval'))))
  .drop('max_time', 'min_time')
  .join(df, ['id', 'time'], 'left')
)

# fill method 1: take last known value

fill_with_last_known = (
  regular_ticks.withColumn(
    'value_interp',
    f.last('value', ignorenulls=True)
    .over(
      w.partitionBy('id')
      .orderBy(
        f.col('time').asc_nulls_last()
      )
    )
  )
)

# fill method 2: linear interpolation

def interpolate(timestamp_col, method={'method': 'fill_zero'}, **kwargs):
    """
    Utility function to interpolate missing values in a timeseries on grouped object, assume group by key is unique id of time series
    also assumes that the timestamp col is regularly spaced already

    schema: output schema of the dataframe
    timestamp_col: column name of timestamp
    method: <map> Supported methods: fill_zero, pad (fill forward), linear, nearest, and other scipy methods

    example usage
    month_series.groupBy(series_id).apply(interpolate(month_series.schema, "month", {'method': 'fill_zero'}))
    series_id here refers to the key/keys of the time series WITHOUT the time column
    """
    def _(pdf):
        pdf.set_index(timestamp_col, inplace=True)
        pdf.sort_index(axis=0, inplace=True)
        if method['method'] == 'fill_zero':
            pdf.fillna(0, inplace=True)
        else:
            pdf.interpolate(**method, inplace=True)
            pdf.ffill(inplace=True)
        pdf.reset_index(drop=False, inplace=True)
        return pdf
    return _

linear_interpolate = (
  regular_ticks.groupBy('id')
  .applyInPandas(
    interpolate('time', {'method': 'linear'}),
    regular_ticks.schema
  )
)

#fill method 3: quadratic interpolation
quadratic_interpolate = (
  regular_ticks.groupBy('id')
  .applyInPandas(
    interpolate('time', {'method': 'quadratic'}),
    regular_ticks.schema
  )
)



