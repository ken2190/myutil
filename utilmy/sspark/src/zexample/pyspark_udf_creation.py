from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
 
 
predict_udf = udf(predict, DoubleType())