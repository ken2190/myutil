from pyspark.sql import Row
from pyspark.sql import HiveContext
from pyspark.sql.functions import udf
from pyspark.context import SparkContext


sc = SparkContext("local", "dict to col")
hc = HiveContext(sc)

data = hc.createDataFrame([Row(user_id=1, app_usage={'snapchat': 2, 'facebook': 10, 'gmail': 1}, active_hours={4: 1, 6: 11, 22: 1}),

                           Row(user_id=2, app_usage={
                               'tinder': 100, 'zoosk': 3, 'hotmail': 2}, active_hours={6: 2, 18: 23, 23: 80}),

                           Row(user_id=3, app_usage={'netflix': 50, 'facebook': 5, 'amazon': 10}, active_hours={10: 4, 19: 6, 20: 55})])

data.show()
rdd = data.select('app_usage', 'user_id').rdd.map(tuple)
rdd.foreach(print)
cols = sorted(list(rdd.map(lambda x: set(x[0].keys())).reduce(
    lambda acc, keys: acc | keys)))
empty_value_fill = 0
new_cols_data = rdd.map(lambda x: [x[1]] + list(map(lambda col: x[0][col]
                                                    if col in x[0] else empty_value_fill, cols))).toDF(['user_id'] + cols)
new_cols_data.show()
