import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as sf
from pyspark.sql import types as st
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import udf, from_json, col, coalesce

import time


## @params: [JOB_NAME]
#args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
#job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "okc-ml", table_name = "second_votes", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
predicate = '(year==2019) and (month==02) and (day==26) and (hour==22)'
partition_keys = ['year','month','day','hour']

times = {}
read_tic = time.time()
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "okc-ml", table_name = "second_votes", 
                                                            transformation_ctx = "datasource0",
                                                            push_down_predicate=predicate)
times['read'] = time.time() - read_tic
datasource0.printSchema()