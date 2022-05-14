from pyspark.sql.functions import array,udf
from pyspark.sql.types import IntegerType

corner_3_udf = udf(is_corner_3, IntegerType())
normal_3_udf = udf(is_normal_3, IntegerType())

df2 = df.withColumn('corner_3', corner_3_udf(array([df.x,df.y])))
df3 = df2.withColumn('normal_3', normal_3_udf(array([df2.x,df2.y,df2.corner_3])))
df4 = df3.withColumn('is_a_3', df3.corner_3 + df3.normal_3) 

df = df4
df.cache()