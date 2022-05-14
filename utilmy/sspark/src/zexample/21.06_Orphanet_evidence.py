from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, udf, when, expr, explode, size, substring, array, regexp_extract, concat_ws,
    collect_set, array_contains
)
from pyspark.sql.types import StringType, IntegerType, TimestampType, StructType
import requests

# establish spark connection
spark = (
    SparkSession.builder
    .master('local[*]')
    .getOrCreate()
)

# Evidence file:
orphanet_file = 'gs://open-targets-pre-data-releases/21.06.5/output/etl/parquet/evidence/sourceId=orphanet/'

orph_df = (
    spark.read.parquet(orphanet_file)
    .filter(col('datasourceId') == 'orphanet')
    .persist()
)

# Get the list of all mapped diseases saved into a file:
(
    orph_df
    .filter(col('diseaseFromSourceMappedId').isNotNull())
    .select(col('diseaseFromSourceMappedId'))
    .distinct()
    .write.format('json').mode('overwrite').option('compression', 'gzip').save('orphanet_diseases.json.gz')
)
 
# Get a look at the diseases:
(
    orph_df
    .filter(col('diseaseFromSourceMappedId').isNotNull())
    .select(col('diseaseFromSourceMappedId'))
    .distinct()
    .show()
)

# Get count of evidence: 5736
orph_df.count()

# Get count of associations: 5736
orph_df.select('diseaseId', 'targetId').distinct().count()

# Get count of associations" 
(
   orph_df
   .groupby('diseaseId', 'targetId')
   .count()
   .filter(col('count') > 1 )
   .show()
)

# Checking one: Orphanet_676|ENSG00000164266
(
    orph_df
    .filter(
        (col('diseaseId') == 'Orphanet_676')
        & (col('targetId') == 'ENSG00000164266')
    )
    .select('diseaseFromSourceId', 'diseaseId', 'targetFromSourceId', 'targetId', 'literature')
    .show(truncate=False)
)

# Get number of diseases:
(
    orph_df
    .select('diseaseId')
    .distinct()
    .count()
)

# Loading all the associations:
assocFile = 'gs://open-targets-pre-data-releases/21.06.5/output/etl/parquet/associationByDatasourceDirect/'
orphanet_only = (
    spark.read.parquet(assocFile)
    .groupby('diseaseId', 'targetId')
    .agg(
        collect_set(col('datasourceId')).alias('sources')
    )
    .filter(
        (array_contains(col('sources'), 'orphanet'))
        & (size(col('sources')) == 1)
    )
    .persist()
)
orphanet_only.count()  # 763
orphanet_only.select('diseaseId').distinct().count()  # 413

# Read TA:
ts = (
    spark.read.option("header",True)
    .csv('gs://ot-team/dsuveges/therapeutic_area/tas_sorted.csv')
    .withColumnRenamed('id', 'TA_id')
    .withColumnRenamed('name', 'TA_label')
)

dis = (
    spark.read
    .option("header",True)
    .option('delimiter', '\t')
    .csv('gs://ot-team/dsuveges/therapeutic_area/orphanet_diseases.tsv')
    .withColumnRenamed('therapeuticArea', 'TA_id')
    .persist()
)
dis_ta = (
    dis
    .join(ts, on='TA_id', how='left')
    .persist()
)

# Get distribution of therapeutic areas:
(
    dis_ta
    .groupby('TA_label')
    .count()
    .orderBy('count')
    .show(40, truncate=False)
)

(
    orphanet_only
    .join(dis_ta, orphanet_only.diseaseId == dis_ta.diseaseFromSourceMappedId, how='left')
    .groupby('TA_label')
    .count()
    .orderBy('count')
    .show(40, truncate=False)
)



@udf(returnType=StringType())
def efo_label_lookup(efo_short):
    url = f'https://www.ebi.ac.uk/ols/api/terms?short_form={efo_short}'
    resp = requests.get(url)

    try:
        response = resp.json()
        label = response['_embedded']['terms'][0]['label']
    except:
        return None

    return label


dis_ta_w_labels = (
    dis_ta
    .withColumn('efo_label', efo_label_lookup(col('diseaseFromSourceMappedId')))
    .persist()
)

# Getting all the diseases that are unique for Orphanet.
# These numbers are re-calculated due to updated evidence
orphanet_only_disease = (
    spark.read.parquet(assocFile)
    .groupby('diseaseId')
    .agg(
        collect_set(col('datasourceId')).alias('sources')
    )
    .filter(
        (array_contains(col('sources'), 'orphanet'))
        & (size(col('sources')) == 1)
    )
    .persist()
)
orphanet_only_disease.count()  # 124

# Adding the labels + the disease label:
orphanet_only_disease_w_ta = (
    orphanet_only_disease
    .withColumn('diseaseLabel', efo_label_lookup(col('diseaseId')))
    .join(dis_ta, orphanet_only_disease.diseaseId == dis_ta.diseaseFromSourceMappedId, how='left')
    .select('diseaseId', 'diseaseLabel', 'TA_label')
)

# Check out the most frequent matches
matches_df = (
    spark.read.parquet('gs://open-targets-pre-data-releases/21.06.5/output/literature/parquet/matches/')
    .filter(col('type') == 'DS')
    .select('pmid', 'keywordId')
)

(
    orphanet_only_disease_w_ta
    .join(matches_df, matches_df.keywordId == orphanet_only_disease_w_ta.diseaseId, how='left')
    .groupby('diseaseId')
    .count()
    .sort('count', ascending=False)
    .show(30)
)
