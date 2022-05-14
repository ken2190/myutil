from pandas import DataFrame
from pyspark.sql import types as t, functions as f

df = DataFrame({'ids': [1, 2, 3], 'words': ['abracadabra', 'hocuspocus', 'shazam']})
sdf = sparkSession.createDataFrame(df)

normalize_word_udf = f.udf(normalize_word, t.StringType())
stops = f.array([f.lit(c) for c in STOPCHARS])

results = sdf.select('ids', normalize_word_udf(f.col('words'), stops).alias('norms'))
results.show()