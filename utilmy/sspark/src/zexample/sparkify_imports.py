import dill

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import avg, col, concat, isnull, desc, explode, lit, min, max, split, udf, count, \
                                    round, when, lag, dense_rank, ceil, floor
from pyspark.sql.types import IntegerType, StructType, StringType, DoubleType, StructField
from pyspark.sql.window import Window

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier,LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, StandardScaler, VectorAssembler, \
                                StringIndexer, IndexToString
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import re
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline