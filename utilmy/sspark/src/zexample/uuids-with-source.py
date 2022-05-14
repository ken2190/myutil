from pyspark.sql import SparkSession
from optparse import OptionParser
from pyspark.sql.types import StringType
import json 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

def get_source(uri):
	uri_suffix = uri.replace('input://seed/', '')
	return uri_suffix.split('/')[1]

def get_inputs_with_source(inputs, source_name):
    has_uuid = (size(col('entity_uris')) > 0)
    has_source = col('uri').like('%seed%')

    uuidUDF = udf(lambda uuid: uuid.replace("entity://", ""), StringType())
    sourceUDF = udf(lambda uri: get_source(uri), StringType())
    
    inputs = inputs.filter(has_uuid & has_source) \
                   .withColumn('source', sourceUDF('uri')) \
                   .select(explode('entity_uris').alias('uuid'), 'source') \
                   .withColumn('uuid', uuidUDF('uuid')) \
                   .select('uuid', 'source')
			
    inputs_with_source = inputs.filter(col('source') == source_name)
    return inputs_with_source.dropDuplicates(['uuid', 'source'])
            
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--parquet", dest="parquet",
                        help="path to base parquet summary inputs")
    parser.add_option("-s", "--source", dest="source",
                        help="name of source to query for")
    parser.add_option("-o", "--output", dest="output",
                        help="output path")
    (options, args) = parser.parse_args()

    spark = SparkSession.builder \
                        .config("mapred.output.compress", "false") \
                        .appName("Get UUIDS") \
                        .getOrCreate()
                
    inputs = spark.read.parquet(options.parquet)
    uuids_with_source = get_inputs_with_source(inputs, options.source)
    uuids_with_source.write.json(options.output)