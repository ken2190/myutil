from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, IntegerType, DateType

#==============Using cast() function
# UDF to process the date column
func = udf(lambda x: datetime.strptime(x, '%d-%m-%Y'), DateType())
df = df \
    .withColumn('colB', func(col('colB'))) \  #to cast the string into DateType we need to specify a UDF in order to process the exact format of the string date.
    .withColumn('colC', col('colC').cast(DoubleType())) \ #change to double
    .withColumn('colD', col('colD').cast(IntegerType())) #change to integer

#==================Using selectExpr() function
df = df.selectExpr(
    'colA',
    'to_date(colB, \'dd-MM-yyyy\') colB',
    'cast(colC as double) colC',
    'cast(colD as int) colD',
)