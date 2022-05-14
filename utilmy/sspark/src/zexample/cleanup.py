import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

spark = SparkSession.builder.appName("datacleanup-pipeline").config("spark.some.config.option", "some-value").getOrCreate()


def run(parameters, parameters_df):
    for param, config in parameters.items():
        if param not in parameters_df.columns:
            continue
        mini = config['limits']['sanity']['min']
        maxi = config['limits']['sanity']['max']
        action = config['limits']['sanity']['action']
        
        def update_frame(val):
            if val is None:
                return None
            val = float(val)
            if val > maxi or val < mini:
                if action == 'drop':
                    return None
            if action == 'clamp':
                val = val > maxi and float(maxi) or val
                val = val < mini and float(mini) or val
                return val
            else:
                return val
        uf = udf(update_frame, FloatType())
        parameters_df = parameters_df.withColumn(param, uf(parameters_df[param]))
    return parameters_df   