import base36
from pyspark.sql.functions import udf
from pyspark.sql.types import *


def base36decoder(x):
    return base36.loads(str(x))

decoder = udf(base36decoder, LongType())


jaja = jaja.select("product_id",decoder("product_id").alias("product_id decode"))

