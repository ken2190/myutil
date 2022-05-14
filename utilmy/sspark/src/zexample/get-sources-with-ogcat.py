from pyspark.sql import SparkSession
from optparse import OptionParser
from pyspark.sql.types import StringType
import json
import pyspark.sql.functions as func
from pyspark.sql.functions import *

def get_source(uri):
        uri_suffix = uri.replace('input://seed/', '')
        return uri_suffix.split('/')[1]

def get_sources_with_original_category(inputs, original_category):
    has_source = col('uri').like('%seed%')
    has_mapped_original_category = array_contains('data.extraction.payload._original_category.mapped', original_category)
    sourceUDF = udf(lambda uri: get_source(uri), StringType())

    source_with_original_category_counts = inputs.filter(has_source & has_mapped_original_category) \
                                                 .withColumn('source', sourceUDF('uri')) \
                                                 .select('source', 'data.extraction.payload._original_category.mapped') \
                                                 .groupBy('source') \
                                                 .count()

    return source_with_original_category_counts

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--parquet", dest="parquet",
                        help="path to base parquet summary inputs")
    parser.add_option("-c", "--category", dest="category",
                        help="name of source to query for")
    parser.add_option("-o", "--output", dest="output",
                        help="output path")
    (options, args) = parser.parse_args()

    spark = SparkSession.builder \
                        .config("mapred.output.compress", "false") \
                        .appName("Get UUIDS") \
                        .getOrCreate()

    inputs = spark.read.parquet(options.parquet)
    sources_with_original_category = get_sources_with_original_category(inputs, options.category)
    sources_with_original_category.write.json(options.output)