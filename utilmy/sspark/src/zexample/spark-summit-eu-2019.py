# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # RDDs

# COMMAND ----------

rdd = sc.parallelize(range(1000), 5)

print(rdd.take(10))
print(rdd.map(lambda x: (x, x * 10)).take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # DataFrames

# COMMAND ----------

df = spark.range(1000)
print(df.limit(10).collect())
df = df.withColumn("col2", df.id * 10)
print(df.limit(10).collect())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Koalas

# COMMAND ----------

import databricks.koalas as ks

kdf = ks.DataFrame(spark.range(1000))
kdf['col2'] = kdf.id * 10
kdf.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Regular UDF

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import udf

@udf
def regularPyUDF(value):
  return value * 10

# COMMAND ----------

df = df.withColumn("col3_udf_", regularPyUDF(df.col2))
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Pandas UDF

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf('integer', PandasUDFType.SCALAR)
def regularPyUDF(pandas_series):
  return pandas_series.multiply(10)

# COMMAND ----------

df = df.withColumn("col3_pandas_udf_", regularPyUDF(df.col2))
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Distributed Small Model Training

# COMMAND ----------

import pyspark.sql.functions as F

raw_dataset = spark.read.format("csv")\
  .option('header', 'true')\
  .option("delimiter", "\t")\
  .option('inferSchema', 'true')\
  .load("dbfs:/databricks-datasets/power-plant/data")\
  .withColumnRenamed("AT", "x0")\
  .withColumnRenamed("V", "x1")\
  .withColumnRenamed("AP", "x2")\
  .withColumnRenamed("RH", "x3")\
  .withColumnRenamed("PE", "y")

# COMMAND ----------

customer_dataset = raw_dataset\
  .withColumn("customer", (F.rand(5) * 15).cast("int"))

# COMMAND ----------

test_dataset = customer_dataset.limit(250).toPandas()
test_dataset.head()

# COMMAND ----------

def train_model(X):
  from sklearn import linear_model
  import pandas as pd
  
  reg = linear_model.LinearRegression()
  X = X.drop("customer", axis=1)
  try:
    x = X.drop("y", axis=1)
    y = X['y']
    reg.fit(x, y)
    print(reg.coef_)
  except:
    return pd.DataFrame([1])
  return pd.DataFrame([0])

# COMMAND ----------

train_model(test_dataset)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import LongType

# set return schema for the DataFrame
model_training_udf = pandas_udf(train_model, 'success long', PandasUDFType.GROUPED_MAP)

# COMMAND ----------

customer_dataset.groupBy("customer").apply(model_training_udf).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Hyperparameter Tuning

# COMMAND ----------

from sklearn import linear_model
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# COMMAND ----------

raw = raw_dataset.toPandas()
x = raw.drop("y", axis=1)
y = raw['y']

# COMMAND ----------

models = [
 Pipeline([
    ("scaler", preprocessing.StandardScaler()),
    ("lr", linear_model.LinearRegression())
  ]),
   Pipeline([
    ("scaler", preprocessing.StandardScaler()),
    ("ridge", linear_model.Ridge())
  ]),
]

parameters = [
  {
    "scaler__with_mean": [True, False],
    "lr__normalize": [True, False],
  },
  {
    "scaler__with_mean": [True, False],
    "ridge__normalize": [True, False],
    "ridge__alpha": [x/10 for x in range(10)]
  },

]

# COMMAND ----------

def train_model_hyperparam(index):
  pipe = models[index[0]]
  params = parameters[index[0]]
#   try:
  model = GridSearchCV(pipe, params, cv=5)
  trained = model.fit(x, y)
#   except:
#     return pd.Series([1])
  return pd.Series([0])

# COMMAND ----------

train_model_hyperparam(pd.Series([0]))

# COMMAND ----------

model_training_udf_2 = pandas_udf(train_model_hyperparam, 'long')

# COMMAND ----------

display(spark.range(len(models)).withColumn("training_result", model_training_udf_2(F.col("id"))))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Hyperparameter Tuning with MLFlow

# COMMAND ----------

from sklearn import linear_model
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# COMMAND ----------

models = [
 Pipeline([
    ("scaler", preprocessing.StandardScaler()),
    ("lr", linear_model.LinearRegression())
  ]),
   Pipeline([
    ("scaler", preprocessing.StandardScaler()),
    ("ridge", linear_model.Ridge())
  ])
]

parameters = [
  {
    "scaler__with_mean": [True, False],
    "lr__normalize": [True, False],
  },
  {
    "scaler__with_mean": [True, False],
    "ridge__normalize": [True, False],
    "ridge__alpha": [x/10 for x in range(10)]
  }
]

# COMMAND ----------

import mlflow
import mlflow.sklearn
experimentName = "SOME EXP NAME"

# COMMAND ----------

def train_model_mlflow(index):
  import mlflow.sklearn
  pipe = models[index[0]]
  params = parameters[index[0]]
  modelType = str(pipe.steps[-1][0])
  
  mlflow.set_experiment(experimentName)
  try:
    model = GridSearchCV(pipe, params, cv=5)
    trained = model.fit(x, y)
    train_mse = mean_squared_error(y, trained.predict(x))
  except:
    return pd.Series([1])
    
  with mlflow.start_run():
    mlflow.sklearn.log_model(trained, modelType + "-pipeline")
    mlflow.sklearn.log_model(trained.best_estimator_, modelType)
    mlflow.log_param("model", modelType)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_param("training", "success")
  return pd.Series([0])

# COMMAND ----------

train_model_mlflow(pd.Series([0]))

# COMMAND ----------

model_training_udf_mlflow = pandas_udf(train_model_mlflow, 'long')

# COMMAND ----------

display(spark.range(len(models)).withColumn("training_result", model_training_udf_mlflow(F.col("id"))))

# COMMAND ----------


