import pyspark.sql.types as T
import pyspark.sql.functions as F

from scipy.stats import nbinom
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# create an arbitrary dataframe
df = spark.createDataFrame([
    [1241, 22, 3, 0.25, 25],
    [1241, 22, 2, 0.3, 40]],
    ['product_id', 'store_id', 'num', 'prob', 'size'])

df.show()
# +----------+--------+---+----+----+
# |product_id|store_id|num|prob|size|
# +----------+--------+---+----+----+
# |      1241|      22|  3|0.25|  25|
# |      1241|      22|  2| 0.3|  40|
# +----------+--------+---+----+----+

def func(arr):
  """
  Function to apply non-spark operations to multiple columns in PySpark DataFrame
  
  :param list arr:
    Column of PySpark.sql.DataFrame with ArrayType, 
    includes 3 columns namely num, prob, size.
  :returns float:
    Summation of cdf multiplied by pmf in a loop
    like: cdf(0) * pmf(3) + 
          cdf(1) * pmf(2) + 
          cdf(2) * pmf(2) + 
          cdf(3) * pmf(0) when num == 3
  """
  amount, size, prob = arr
  amount = int(amount)
  output = [nbinom.cdf(i, size, prob) * nbinom.pmf(amount-i-1, size, prob) 
            for i in range(amount)]
  return float(sum(output))

udf_func = F.udf(lambda x: func(x), T.FloatType())

df = df.withColumn('arr', F.array(F.col('num'), F.col('size'), F.col('prob')))
df = df.withColumn('new', udf_func(F.col('arr'))).show()

# +----------+--------+---+----+----+-----------------+-------------+
# |product_id|store_id|num|prob|size|              arr|          new|
# +----------+--------+---+----+----+-----------------+-------------+
# |      1241|      22|  3|0.25|  25|[3.0, 25.0, 0.25]|5.9613233E-28|
# |      1241|      22|  2| 0.3|  40| [2.0, 40.0, 0.3]|   8.4252E-41|
# +----------+--------+---+----+----+-----------------+-------------+
                  
