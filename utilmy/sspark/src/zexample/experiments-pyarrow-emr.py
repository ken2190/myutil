
sudo pip install PyArrow

./pyspark --master yarn --num-executors 2

from pyspark.sql.functions import rand
df = spark.range(1 << 22).toDF("id").withColumn("x", rand())

from pyspark.sql.functions import udf
@udf('double')
def plus_one(v):
      return v + 1

from pyspark.sql.functions import *
df.repartition(2).withColumn('id2', plus_one(df.id)).agg(count("id2")).show()

from pyspark.sql.functions import pandas_udf, PandasUDFType
@pandas_udf('double', PandasUDFType.SCALAR)
def pandas_plus_one(v):
    return v + 1
  
df.repartition(2).withColumn('id2', pandas_plus_one(df.id)).agg(count("id2")).show()

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
