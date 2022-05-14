# load pandas, sklearn, and pyspark types and functions
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *