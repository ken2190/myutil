from pyspark.sql import SparkSession
from optparse import OptionParser
from pyspark.sql.types import StringType, ArrayType
import json 
import pyspark.sql.functions as func
from pyspark.sql.functions import *

def get_source(uri):
	uri_suffix = uri.replace('input://seed/', '')
	return uri_suffix.split('/')[1]

def get_inputs_with_source(inputs, source_name):
    has_uuid = (size(col('entity_uris')) > 0)
    has_source = col('uri').like('%seed%')
    has_mapped = col('data.extraction.payload._original_category.mappings').isNotNull()

    uuidUDF = udf(lambda uuid: uuid.replace("entity://", ""), StringType())
    sourceUDF = udf(lambda uri: get_source(uri), StringType())
    
    inputs = inputs.filter(has_uuid & has_source & has_mapped) \
                   .withColumn('source', sourceUDF('uri')) \
                   .select(explode('entity_uris').alias('uuid'), 'source', 'data.extraction.payload._original_category.mappings') \
                   .withColumn('uuid', uuidUDF('uuid')) \
                   .select('uuid', 'source', 'mappings')
			
    inputs_with_source = inputs.filter(col('source') == source_name))
    return inputs_with_source.dropDuplicates(['uuid', 'source'])
            
def get_stable_entities(entities):
    stable_view = 'view.places_' + cc + '_stable'
    emptyarray = udf(lambda x: [], ArrayType(StringType()))
  
    stable_entities = entities.filter(array_contains('exportedViews', stable_view)) \
                            .select('uuid', 'payload.name', 'payload.category_ids')
    
    with_category = stable_entities.filter(col('category_ids').isNotNull())
    without_category = stable_entities.filter(col('category_ids').isNull()) \
                                    .withColumn('category_ids', emptyarray('uuid'))
  
    all_stable_entities = with_category.union(without_category)
    return all_stable_entities

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
                
    inputs = spark.read.parquet(options.parquet + '/summary_inputs')
    entities = spark.read.parquet(options.parquet + '/entities')
    
    stable_entities = get_stable_entities(entities)
    inputs_with_source = get_inputs_with_source(inputs, options.source)
    
    df = stable_entities.join(inputs_with_source, 'uuid', 'left_outer') \
                        .select('uuid', stable_entities.name, stable_entities.category_ids, inputs_with_source.source, inputs_with_source.mappings)
      
    df.write.json(options.output)
