# udfs are applied to col elements, not to cols
# but they take col as args (pyspark.sql.Column)
# and return (pyspark.sql.types)
from pyspark.sql import functions as F

>>> def f(c1, c2):
      return str(c1) + str(c2)
>>> fu = F.udf(f, StringType())
>>> df = spark.createDataFrame([(1, 'a'), (1, 'b'), (2, 'd')], ['c1', 'c2'])
>>> df.withColumn('test', fu(df.c1, df.c2)).show()
+---+---+----+
| c1| c2|test|
+---+---+----+
|  1|  a|  1a|
|  1|  b|  1b|
|  2|  d|  2d|
+---+---+----+


## MAP ##
# mapping occurs with a withColumn, see UDF
>>> @F.udf(returnType=T.StringType())
    def f(c1, c2):
      return str(c1) + str(c2)
>>> df = spark.createDataFrame([(1, '123'), (1, '90'), (2, '45')], ['c1', 'c2'])
>>> df.withColumn('test', f(df.c1, df.c2)).show()
+---+---+----+
| c1| c2|test|
+---+---+----+
|  1|123|1123|
|  1| 90| 190|
|  2| 45| 245|
+---+---+----+




## FILTER ##

@F.udf(T.BooleanType())
def g(c1, c2):
      return int(c1) > 1 & int(c2) % 2 == 0
df = spark.createDataFrame([(1, '123'), (1, '90'), (2, '45')], ['c1', 'c2'])
df.filter(g(df.c1, df.c2)).show()
'''
+---+---+
| c1| c2|
+---+---+
|  1| 90|
+---+---+
'''

# CURRYING
# la curryfication désigne la transformation d'une fonction à plusieurs
# arguments en une fonction à un argument qui retourne une fonction sur
# le reste des arguments.