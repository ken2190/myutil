import requests
import time
import json
import requests

import pyspark.sql
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import col, regexp_extract, collect_set, count, udf
spark = (
    pyspark.sql.SparkSession
    .builder
    .master("local")
    .getOrCreate()
)

phewas_file = 'gs://otar000-evidence_input/PheWAS/json/phewas_catalog-2021-06-08.json.gz' # Using the old dataset to make sure we have the target id already added to the evidence
genetics_file = 'gs://otar000-evidence_input/Genetics_portal/json/genetics_portal-2021-06-21'


# Reading data: 192,448
phewas_df = (
    spark.read.json(phewas_file)
    .select('diseaseFromSource', 'diseaseFromSourceMappedId', 'variantRsId', 'targetFromSourceId', 'targetFromSource')
    .distinct()
    .persist()
)

# How many distinct genes do we have:
phewas_df.select('targetFromSourceId').distinct().count()

# Genetics file: 580,130
genetics_df = (
    spark.read.json(genetics_file)
    .select('diseaseFromSource', 'diseaseFromSourceMappedId', 'variantRsId', 'targetFromSourceId')
    .distinct()
    .persist()
)

# 157,505  unique association:
phewas_assoc = (
    phewas_df
    .select('diseaseFromSourceMappedId', 'targetFromSourceId')
    .distinct()
    .persist()
)

# 279,585 unique associations:
genetics_assoc =( 
    genetics_df
    .select('diseaseFromSourceMappedId', 'targetFromSourceId')
    .filter(col('diseaseFromSourceMappedId').isNotNull())
    .distinct()
    .persist()
)

# Merging:
(
   phewas_assoc
   .join(genetics_assoc, how='left_anti', on=['diseaseFromSourceMappedId', 'targetFromSourceId'])
   .count()
)

(
    phewas_assoc.select('diseaseFromSourceMappedId').distinct()
    .join(genetics_assoc.select('diseaseFromSourceMappedId').distinct(), how='left_anti', on='diseaseFromSourceMappedId')
    .count()
)
# Oh no, there are some strange rows:
# In [11]: genetics_df.show()
# +--------------------+-------------------------+-----------+------------------+ 
# |   diseaseFromSource|diseaseFromSourceMappedId|variantRsId|targetFromSourceId|
# +--------------------+-------------------------+-----------+------------------+
# |Serum metabolite ...|              EFO_0005653|   rs174561|   ENSG00000221968|
# |              Weight|              EFO_0004338|  rs6597975|   ENSG00000177685|
# |High light scatte...|              EFO_0007986|  rs4929915|   ENSG00000166452|
# |                null|                     null| rs73230336|   ENSG00000135097|

# 20,501
null_disease = (
    spark.read.json(genetics_file)
    .select('diseaseFromSourceMappedId', 'targetFromSourceId', 'variantId', 'variantRsId', 'studyId', 'studyId')
    .filter(col('diseaseFromSourceMappedId').isNull())
    .distinct()
    .persist()
)

# I guess these are all ukbb studies:
null_disease.select('studyId').distinct().show()

# In [20]: null_disease.select('studyId').distinct().count()
# Out[20]: 318 
null_disease.select('studyId').filter(col('studyId').startswith('GCST')).distinct().count()
# Something is wrong:
# In [22]: null_disease.select('studyId').filter(col('studyId').startswith('GCST')).distinct().show()
# +----------+                                                                    
# |   studyId|
# +----------+
# |GCST010653|
# |GCST011378|
# |GCST010729|
# +----------+

STUDY_FILE="gs://ot-team/dsuveges/tep/studies.parquet"

study_df = (
    spark.read.parquet(STUDY_FILE)
    .select('study_id', 'pmid', 'trait_reported', 'trait_efos', 'trait_category', 'has_sumstats')
    .distinct().persist()
)

(
    null_disease.select('studyId').filter(col('studyId').startswith('GCST')).distinct()
    .join(study_df, null_disease.studyId == study_df.study_id, how='inner')
    .show()
)

# out of 1,101 diseases -> 510 were unique for PheWAS
(
    phewas_assoc.select('diseaseFromSourceMappedId').distinct()
    .join(genetics_assoc.select('diseaseFromSourceMappedId').distinct(), how='left_anti', on='diseaseFromSourceMappedId')
    .count()
)

# out of 1,746 targets -> 61 were unique for PheWAS
(
    phewas_assoc.select('targetFromSourceId').distinct()
    .join(genetics_assoc.select('targetFromSourceId').distinct(), how='left_anti', on='targetFromSourceId')
    .count()
)

# How about associations:
# out of 157,505 unique associations -> 156,174 (only 1,400 shared)
phewas_assoc.select('targetFromSourceId', 'diseaseFromSourceMappedId').distinct().count()
(
    phewas_assoc.select('targetFromSourceId', 'diseaseFromSourceMappedId').distinct()
    .join(genetics_assoc.select('targetFromSourceId', 'diseaseFromSourceMappedId').distinct(), how='left_anti', on=['targetFromSourceId', 'diseaseFromSourceMappedId'])
    .distinct()
    .count()
)

# We have to compare data with ontology exansion... 
(
    spark.read.parquet('gs://open-targets-data-releases/21.06/output/etl/parquet/associationByDatasourceIndirect')
    .show()
)