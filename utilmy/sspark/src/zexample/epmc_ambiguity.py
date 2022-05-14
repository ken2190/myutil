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

@udf(ArrayType(StringType()))

def ols_lookup(keywords):
    labels = []
    for keyword in keywords:
        response = requests.get(f'https://www.ebi.ac.uk/ols/api/terms?short_form={keyword}').json()
        label = response['_embedded']['terms'][0]['label']
        labels.append(label)

    return labels


# EPMC dafile 
(
    spark.read.parquet('gs://open-targets-pre-data-releases/21.06.5/output/literature/parquet/matches/')
    .limit(100_000)
    .filter(
        (col('type') == 'DS')
        & (col('ismapped') == True)
    )
    .select(col('label'), col('keywordId'))
    .distinct()
    .groupby(col('label'))
    .agg(
        collect_set(col('keywordId')).alias('keywords'),
        count(col('keywordId')).alias('count')
    )
    .filter(col('count') >= 3)
    .limit(20)
    .withColumn('labels', ols_lookup(col('keywords')))
    .show(truncate=False)
)

# +------------------------+-----------------------------------------------------------+-----+------------------------------------------------------------------------------------------------------+
# |label                   |keywords                                                   |count|labels                                                                                                |
# +------------------------+-----------------------------------------------------------+-----+------------------------------------------------------------------------------------------------------+
# |ACE                     |[Orphanet_946, Orphanet_36, EFO_0007129]                   |3    |[Acrocephalosyndactyly, Acrocallosal syndrome, acute chest syndrome]                                  |
# |ACS                     |[Orphanet_946, Orphanet_36, EFO_0007129]                   |3    |[Acrocephalosyndactyly, Acrocallosal syndrome, acute chest syndrome]                                  |
# |APE                     |[EFO_0004533, MONDO_0017278, Orphanet_320]                 |3    |[alkaline phosphatase measurement, autoimmune polyendocrinopathy, Apparent mineralocorticoid excess]  |
# |CA                      |[Orphanet_209908, EFO_0009860, EFO_0000311]                |3    |[Childhood apraxia of speech, chromosomal aberration frequency, cancer]                               |
# |CAS                     |[Orphanet_209908, EFO_0009860, EFO_0000311]                |3    |[Childhood apraxia of speech, chromosomal aberration frequency, cancer]                               |
# |CHS                     |[Orphanet_2686, Orphanet_167, EFO_0009205]                 |3    |[Cyclic neutropenia, Chédiak-Higashi syndrome, Corpuscular Hemoglobin Content]                        |
# |Colon cancer            |[EFO_0000365, MONDO_0021063, EFO_1001950, EFO_0004288]     |4    |[colorectal adenocarcinoma, malignant colon neoplasm, colon carcinoma, colonic neoplasm]              |
# |DM                      |[EFO_0010133, EFO_0000398, EFO_0000400]                    |3    |[diabetic maculopathy, dermatomyositis, diabetes mellitus]                                            |
# |ES                      |[EFO_0000174, MONDO_0017387, EFO_0009200]                  |3    |[Ewings sarcoma, epithelioid sarcoma, Eisenmenger syndrome]                                           |
# |HP                      |[EFO_1000299, MONDO_0015540, Orphanet_79430, MONDO_0017853]|4    |[Hyperplastic Polyp, hemophagocytic syndrome, Hermansky-Pudlak syndrome, hypersensitivity pneumonitis]|
# |Hp                      |[EFO_1000299, MONDO_0015540, Orphanet_79430, MONDO_0017853]|4    |[Hyperplastic Polyp, hemophagocytic syndrome, Hermansky-Pudlak syndrome, hypersensitivity pneumonitis]|
# |MA                      |[Orphanet_562, EFO_1001806, EFO_1001037]                   |3    |[McCune-Albright syndrome, macrophage activation syndrome, meconium aspiration syndrome]              |
# |MÃ                      |[Orphanet_562, EFO_1001806, EFO_1001037]                   |3    |[McCune-Albright syndrome, macrophage activation syndrome, meconium aspiration syndrome]              |
# |MDS                     |[EFO_0000198, Orphanet_531, Orphanet_565]                  |3    |[myelodysplastic syndrome, Miller-Dieker syndrome, Menkes disease]                                    |
# |Non-melanoma Skin Cancer|[EFO_0009260, MONDO_0002898, EFO_0009259]                  |3    |[non-melanoma skin carcinoma, skin cancer, skin carcinoma]                                            |
# |Oral cancer             |[MONDO_0023644, EFO_0003868, EFO_0003871]                  |3    |[lip and oral cavity carcinoma, mouth neoplasm, tongue neoplasm]                                      |
# |Pancreatic Cancer       |[EFO_0002618, EFO_1000359, EFO_0003860]                    |3    |[pancreatic carcinoma, Malignant Pancreatic Neoplasm, pancreatic neoplasm]                            |
# |Pancreatic cancer       |[EFO_0002618, EFO_1000359, EFO_0003860]                    |3    |[pancreatic carcinoma, Malignant Pancreatic Neoplasm, pancreatic neoplasm]                            |
# |Pinealomas              |[MONDO_0021232, MONDO_0003249, EFO_1000475]                |3    |[pineal body neoplasm, pineal gland cancer, Pineoblastoma]                                            |
# |Testicular cancer       |[EFO_0005088, EFO_0004281, MONDO_0003510]                  |3    |[testicular carcinoma, testicular neoplasm, malignant testicular germ cell tumor]                     |
# +------------------------+-----------------------------------------------------------+-----+------------------------------------------------------------------------------------------------------+


# Get schema for the match data:
spark.read.parquet('gs://open-targets-pre-data-releases/21.06.5/output/literature/parquet/matches/').printSchema()
# root
#  |-- pmid: string (nullable = true)
#  |-- pmcid: string (nullable = true)
#  |-- pubDate: string (nullable = true)
#  |-- date: date (nullable = true)
#  |-- year: integer (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- day: integer (nullable = true)
#  |-- organisms: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- section: string (nullable = true)
#  |-- text: string (nullable = true)
#  |-- endInSentence: long (nullable = true)
#  |-- label: string (nullable = true)
#  |-- sectionEnd: long (nullable = true)
#  |-- sectionStart: long (nullable = true)
#  |-- startInSentence: long (nullable = true)
#  |-- type: string (nullable = true)
#  |-- keywordId: string (nullable = true)
#  |-- isMapped: boolean (nullable = true)

# Get an example:
spark.read.parquet('gs://open-targets-pre-data-releases/21.06.5/output/literature/parquet/matches/').show(1,vertical=True, truncate=False)
# -RECORD 0----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  pmid            | 33552545                                                                                                                                                                      
#  pmcid           | PMC7850001                                                                                                                                                                    
#  pubDate         | 2020-12-01                                                                                                                                                                    
#  date            | 2020-12-01                                                                                                                                                                    
#  year            | 2020                                                                                                                                                                          
#  month           | 12                                                                                                                                                                            
#  day             | 1                                                                                                                                                                             
#  organisms       | []                                                                                                                                                                            
#  section         | abstract                                                                                                                                                                      
#  text            | Transverse skin incision may be preferable to vertical skin incision at cesarean delivery in pregnant patients with obesity as it may be associated with a lower rate of WCs. 
#  endInSentence   | 123                                                                                                                                                                           
#  label           | obesity                                                                                                                                                                       
#  sectionEnd      | 1001                                                                                                                                                                          
#  sectionStart    | 828                                                                                                                                                                           
#  startInSentence | 116                                                                                                                                                                           
#  type            | DS                                                                                                                                                                            
#  keywordId       | EFO_0001073                                                                                                                                                                   
#  isMapped        | true                                                                                                                                                                          
