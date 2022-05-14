import inspect
from typing import Callable, List
import pandas as pd
from pyspark.sql import DataFrame, Row, column
from pyspark.sql.functions import lit, pandas_udf, PandasUDFType, array
from pyspark.sql.types import FloatType
sdf = sc.parallelize([
    Row(price=1, quantity=1, shipping_cost_collected=0, costs=[0.0,1.0]),
    Row(price=4, quantity=2, shipping_cost_collected=2, costs=[0.0,2.0]),
    Row(price=7, quantity=3, shipping_cost_collected=0, costs=[3.0,1.0])
]).toDF()
sdf.toPandas()
