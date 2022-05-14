from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, udf, when, expr, explode, size, substring, array, regexp_extract, concat_ws,
    collect_set, array_contains
)

from pyspark.sql.functions import sum as ps_sum
from pyspark.sql.types import StringType, IntegerType, TimestampType, StructType
import requests

# establish spark connection
spark = (
    SparkSession.builder
    .master('local[*]')
    .getOrCreate()
)

new_file = 'gs://open-targets-pre-data-releases/21.06.5/output/etl/parquet/evidence/sourceId=ot_genetics_portal/'
old_file = 'gs://open-targets-data-releases/21.04/output/etl/parquet/evidence/sourceId=ot_genetics_portal/'

# Read both files:
new_df = spark.read.parquet(new_file).persist()
old_df = spark.read.parquet(old_file).persist()

# Get new studies:
studies = new_df.select('studyId').distinct()
old_studies = old_df.select('studyId').distinct()

old_associations = old_df.select('diseaseId', 'targetId').distinct().persist()

new_studies = (
    studies
    .join(old_studies, on='studyId', how='left_anti')
)

print(f'New studies in the data: {new_studies.count()}')

# Filter new data for new studies:
new_data = (
    new_df
    .join(new_studies, on='studyId', how='inner')
    .persist()
)

print(f'Evidence in the newly added studies: {new_data.count()}') # 220k
print(f'Distribution of sources in the new studies:') # 
print(new_data.select('studyId','projectId').distinct().groupby('projectId').count().show())

# +---------+-----+
# |projectId|count|
# +---------+-----+
# |    NEALE|    1|
# |    SAIGE|    2|
# |  FINNGEN| 1145|
# |     GCST| 4065|
# +---------+-----+

print(f"Number of association: {new_data.select('diseaseId','targetID').distinct().count()}")  # 120k

# Get evidence for new associations:
evidence_new_assoc = (
    new_data
    .join(old_associations, on=['diseaseId', 'targetId'], how='left_anti')
    .persist()
)
print(f"Evidence count for new evidence: {evidence_new_assoc.count()}")
print(f"Evidence count for new evidence: {evidence_new_assoc.select('diseaseId', 'targetId').distinct().count()}")

(
    evidence_new_assoc
    .select('diseaseId', 'targetId', 'projectId')
    .distinct()
    .groupby('diseaseid', 'targetId')
    .agg(
        collect_set(col('projectId')).alias('projects')
    )
    .groupby('projects')
    .count()
    .show()
)

# +----------------+-----+                                                        
# |        projects|count|
# +----------------+-----+
# |         [NEALE]|    2|
# |         [SAIGE]|   60|
# |       [FINNGEN]| 8156|
# |[FINNGEN, SAIGE]|   35|
# | [GCST, FINNGEN]|  109|
# |          [GCST]|75435|
# +----------------+-----+

poscon = (
    spark.read.option('header', True)
    .csv('gs://ot-team/dsuveges/positive_controls/abbvie_pharmaprojects_2018_mapped.csv')
    .withColumnRenamed('ensembl_id', 'targetId')
    .withColumnRenamed('id', 'diseaseId')
    .withColumnRenamed('lApprovedUS.EU', 'lApprovedUSEU')
    .withColumn('isApproved', when(col('lApprovedUSEU') == 'TRUE', True).othewise(False))
    .drop(col('lApprovedUSEU'))
    .persist()
)

# Is any of the poscons overlap with the new gwas data:
new_poscon = (
    evidence_new_assoc
    .select('diseaseId','studyId','targetId','projectId')
    .distinct()
    .join(poscon, on=['diseaseId', 'targetId'], how='inner')
    .persist()
)

# Get details:
print(f'Number of new associations with poscon: {new_poscon.count()}')  # 290
print(f"Number with approved assoc: {new_poscon.filter(col('isApproved') == 'TRUE').count()}") # 28

# What do we have here:
(
    new_poscon
    .filter(col('isApproved') == 'TRUE')
    .groupby('projectId')
    .count().show()
)

## Approved poscon:
# +---------+-----+
# |projectId|count|
# +---------+-----+
# |  FINNGEN|    8|
# |     GCST|   20|
# +---------+-----+

(
    new_poscon
    .groupby('projectId')
    .count().show()
)

## Poscon discribution:
# +---------+-----+
# |projectId|count|
# +---------+-----+
# |  FINNGEN|   68|
# |     GCST|  141|
# +---------+-----+

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

@udf(returnType=StringType())
def symbol_lookup(ensembl_id):
    url = f'http://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json'
    resp = requests.get(url)

    try:
        response = resp.json()
        label = response['display_name']
    except:
        return None

    return label


# Let's see some examples from the approved:
new_poscon_mapped = (
    new_poscon
    .withColumn('diseaseLabel', efo_label_lookup(col('diseaseId')))
    .withColumn('targetLable', symbol_lookup(col('targetId')))
    .persist()
)

# All good.