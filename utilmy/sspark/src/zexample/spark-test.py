from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType
import hashlib

csvFile="test.csv"

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(csvFile)

colFile="cols-list.txt"
colsToHash = sc.textFile(colFile).collect()

def hash(str):
    import hashlib
    m = hashlib.md5()
    m.update(""+str)
    return m.hexdigest()

udf = UserDefinedFunction(lambda x: hash(x), StringType())
new_df = df.select(*[udf(column).alias(column) if column in colsToHash else column for column in df.columns])

csvFileOut="test-out4.csv"
new_df.repartition(1).write.format('com.databricks.spark.csv').save(csvFileOut,header = 'true')

