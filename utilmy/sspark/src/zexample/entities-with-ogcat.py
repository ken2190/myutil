from pyspark.sql import SparkSession
from optparse import OptionParser
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
import json 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

def get_source(uri):
	uri_suffix = uri.replace('input://seed/', '')
	return uri_suffix.split('/')[1]

def restructure_inputs(inputs):
    has_uuid = (size(col('entity_uris')) > 0)
    has_source = col('uri').like('%seed%')

    uuidUDF = udf(lambda uuid: uuid.replace("entity://", ""), StringType())
    sourceUDF = udf(lambda uri: get_source(uri), StringType())
    
    inputs = inputs.filter(has_uuid & has_source) \
                   .withColumn('source', sourceUDF('uri')) \
                   .select(explode('entity_uris').alias('uuid'), 'data.extraction.payload._original_category.mapped', \
                           'source', col('details.wiki.rawInputTimestamp').alias('timestamp'), \
                           'data.extraction.payload._original_category.unmapped', 'data.extraction.payload.category_ids') \
                   .withColumn('uuid', uuidUDF('uuid')) \
                   .select('uuid', 'source', 'timestamp', 'mapped', 'unmapped', 'category_ids')
    
    w = Window.partitionBy('uuid', 'source').orderBy((inputs['source']).desc())
    most_recent_input = (func.max(inputs['timestamp']).over(w))

    most_recent_inputs = inputs.select('uuid', 'source', 'timestamp', 'mapped', 'unmapped', 'category_ids', most_recent_input.alias('most_recent')) \
				.filter(col('timestamp') == col('most_recent')) \
				.drop('timestamp', 'most_recent') \
				.dropDuplicates(['uuid', 'source'])
    
    #### filter by ogcat
    most_recent_inputs_with_ogcat = most_recent_inputs.filter(array_contains('mapped', INSERT_OG_CAT_HERE))
    return most_recent_inputs_with_ogcat
            
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--parquet", dest="parquet",
                        help="path to base parquet summary inputs")
    parser.add_option("-o", "--output", dest="output",
                        help="output path")
    (options, args) = parser.parse_args()

    spark = SparkSession.builder \
                        .config("mapred.output.compress", "false") \
                        .appName("Get UUIDS") \
                        .getOrCreate()
                
    inputs = spark.read.parquet(options.base)
    inputs_with_ogcat = restructure_inputs(inputs)
    inputs_with_ogcat.write.json(options.output)
