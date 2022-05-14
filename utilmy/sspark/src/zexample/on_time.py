Dataset:df_ts_list
+--------------------+
|             ts_list|
+--------------------+
|[1477411200, 1477...|
|[1477238400, 1477...|
|[1477022400, 1477...|
|[1477224000, 1477...|
|[1477256400, 1477...|
|[1477346400, 1476...|
|[1476986400, 1477...|
|[1477321200, 1477...|
|[1477306800, 1477...|
|[1477062000, 1477...|
|[1477249200, 1477...|
|[1477040400, 1477...|
|[1477090800, 1477...|
+--------------------+


Pyspark UDF:

>>> def on_time(ts_list):
...     import sys
...     import os
...     sys.path.append('/usr/lib/python2.7/dist-packages')
...     os.system("sudo apt-get install python-numpy -y")
...     import numpy as np
...     import datetime
...     import time
...     from datetime import timedelta
...     ts = np.array(ts_list)
...     if ts.size == 0:
...             count = 0
...             duration = 0
...             st = time.mktime(datetime.now())
...             ymd = str(datetime.fromtimestamp(st).date())
...     else:
...             ts.sort()
...             one_tag = []
...             start = float(ts[0])
...             for i in range(len(ts)):
...                     if i == (len(ts)) - 1:
...                             end = float(ts[i])
...                             a_round = [start, end]
...                             one_tag.append(a_round)
...                     else:
...                             diff = (datetime.datetime.fromtimestamp(float(ts[i+1])) - datetime.datetime.fromtimestamp(float(ts[i])))
...                             if abs(diff.total_seconds()) > 3600:
...                                     end = float(ts[i])
...                                     a_round = [start, end]
...                                     one_tag.append(a_round)
...                                     start = float(ts[i+1])
...             one_tag = [u for u in one_tag if u[1] - u[0] > 300]
...             count = int(len(one_tag))
...             duration = int(np.diff(one_tag).sum())
...             ymd = str(datetime.datetime.fromtimestamp(time.time()).date())
...     return {'count':count,'duration':duration, 'ymd':ymd}
  
  
  
  
  pyspark code:
  
  
>>> on_time=udf(on_time, MapType(StringType(),StringType()))
>>> df_ts_list.withColumn("one_tag", on_time("ts_list")).select("one_tag").show()
  
  
  
  Error:
  
  Caused by: org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/usr/lib/spark/python/pyspark/worker.py", line 172, in main
    process()
  File "/usr/lib/spark/python/pyspark/worker.py", line 167, in process
    serializer.dump_stream(func(split_index, iterator), outfile)
  File "/usr/lib/spark/python/pyspark/worker.py", line 106, in <lambda>
    func = lambda _, it: map(mapper, it)
  File "/usr/lib/spark/python/pyspark/worker.py", line 92, in <lambda>
    mapper = lambda a: udf(*a)
  File "/usr/lib/spark/python/pyspark/worker.py", line 70, in <lambda>
    return lambda *a: f(*a)
  File "<stdin>", line 27, in on_time
  File "/usr/lib/spark/python/pyspark/sql/functions.py", line 39, in _
    jc = getattr(sc._jvm.functions, name)(col._jc if isinstance(col, Column) else col)
AttributeError: 'NoneType' object has no attribute '_jvm'

  
  
