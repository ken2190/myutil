# Pre Spark 2.1, use the tag 'pre-2.1'
spark._jvm.com.ing.wbaa.spark.udf.ValidateIBAN.registerUDF(spark._jsparkSession)
# Spark 2.1+, use the tag '2.1+'
from pyspark.sql.types import BooleanType
sqlContext.registerJavaFunction("validate_iban", "com.ing.wbaa.spark.udf.ValidateIBAN", BooleanType())
# Spark 2.3+ use the tag '2.1+'
from pyspark.sql.types import BooleanType
spark.udf.registerJavaFunction("validate_iban", "com.ing.wbaa.spark.udf.ValidateIBAN", BooleanType())

# Use your UDF!
spark.sql("""SELECT validate_iban('NL20INGB0001234567')""").show()