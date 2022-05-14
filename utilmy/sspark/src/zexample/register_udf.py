from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

spark_train_and_test_udf = udf(train_and_test_udf, StringType())
