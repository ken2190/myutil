import joblib # To Pickel Trained model file 
import numpy as np # To create random data
import pandas as pd # To operate on data in Python Process
from sklearn.linear_model import LinearRegression # To train Linear Regression models

from pyspark.sql.functions import pandas_udf, PandasUDFType # Pandas UDF functions to call Python processes from spark
from pyspark.sql.types import DoubleType, StringType, ArrayType # Data types to capture reurn at Spark End