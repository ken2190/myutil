import os
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameReader
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf


#Functions to mask the data columns based on Account ID or SWIFT BIC
def update_STREET_ADDRESS(ACCOUNT_ID):
    return "Street Address for "+ACCOUNT_ID


def update_SECONDARY_ADDRESS(ACCOUNT_ID):
    return "Secondary Address for "+ACCOUNT_ID

def update_POSTAL_CODE(ACCOUNT_ID):
    return "Postal Code for "+ACCOUNT_ID

def update_CITY(ACCOUNT_ID):
    return "City for "+ACCOUNT_ID
    
def update_ZIP_CODE(ACCOUNT_ID):
    return "Zip Code for "+ACCOUNT_ID

def update_SWIFT_ADDR(SWIFT_ADDR):
    return SWIFT_ADDR[:-2]+"XXXX"

def update_TEL_NUM(ACCOUNT_ID):
    return "Tel No for "+ACCOUNT_ID
    
def update_EMAIL_ADDR(ACCOUNT_ID):
    return "Email ID for "+ACCOUNT_ID
    
def update_CNTCT_PRSN(ACCOUNT_ID):
    return "Contact Person for "+ACCOUNT_ID
    
def update_CMPNY_NAME(ACCOUNT_ID):
    return "Company Name "+ACCOUNT_ID
    
def update_FAX_NUM(ACCOUNT_ID):
    return "Fax Num "+ACCOUNT_ID

#Create Spark Context & Session Object
conf = SparkConf().setAppName('Simple App')
sc = SparkContext("local", "Simple App")
spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
sqlContext = SQLContext(sc)

# Path for spark source folder
os.environ['SPARK_HOME']="C:/Users/USER1/rcs/spark-2.1.0-bin-hadoop2.6"
os.environ['SPARK_CLASSPATH']="C:/Users/USER1/Documents/python/test/100_script_30_day_challenge/pyspark/postgresql-42.1.1.jre6.jar"
# Append pyspark  to Python Path
sys.path.append("C:/Users/USER1/rcs/spark-2.1.0-bin-hadoop2.6/python")
sys.path.append("C:/Users/USER1/rcs/spark-2.1.0-bin-hadoop2.6/python/lib/py4j-0.10.4-src.zip")


spark = SparkSession.builder\
    .master('local[*]')\
    .appName('My App')\
    .config('spark.sql.warehouse.dir', 'file:///C:/temp')\
    .getOrCreate()


	
#Convert RDD to DataFrame
cols = ('ACCOUNT_ID','STREET_ADDRESS','SECONDARY_ADDRESS','POSTAL_CODE','CITY','COUNTRY','COUNTRY_CODE',
        'ZIP_CODE','SWIFT_ADDR','TEL_NUM','EMAIL_ADDR','CNTCT_PRSN','CMPNY_NAME','FAX_NUM')



# Define JDBC properties for DB Connection
url = "jdbc:postgresql://localhost/postgres"
properties = {
    "user": "pridash4",
    "driver": "org.postgresql.Driver"
}


#Read the BIC & Account Data from DB
df = DataFrameReader(sqlContext).jdbc(
    url=url, table='test_bics1', properties=properties
)

val1 = df.count()
print val1

df.registerTempTable("test_bics1")

#Mask the Data Colums
sqlContext.udf.register("update_STREET_ADDRESS_udf",update_STREET_ADDRESS,StringType())
sqlContext.udf.register("update_SECONDARY_ADDRESS_udf",update_SECONDARY_ADDRESS,StringType())
sqlContext.udf.register("update_POSTAL_CODE_udf",update_POSTAL_CODE,StringType())
sqlContext.udf.register("update_CITY_udf",update_CITY,StringType())
sqlContext.udf.register("update_ZIP_CODE_udf",update_ZIP_CODE,StringType())
sqlContext.udf.register("update_SWIFT_ADDR_udf",update_SWIFT_ADDR,StringType())
sqlContext.udf.register("update_TEL_NUM_udf",update_TEL_NUM,StringType())
sqlContext.udf.register("update_EMAIL_ADDR_udf",update_EMAIL_ADDR,StringType())
sqlContext.udf.register("update_CNTCT_PRSN_udf",update_CNTCT_PRSN,StringType())
sqlContext.udf.register("update_CMPNY_NAME_udf",update_CMPNY_NAME,StringType())
sqlContext.udf.register("update_FAX_NUM_udf",update_FAX_NUM,StringType())


df1 = sqlContext.sql("select ACCOUNT_ID,update_STREET_ADDRESS_udf(ACCOUNT_ID) as STREET_ADDRESS,update_SECONDARY_ADDRESS_udf(ACCOUNT_ID) as SECONDARY_ADDRESS,update_POSTAL_CODE_udf(ACCOUNT_ID) as POSTAL_CODE,update_CITY_udf(ACCOUNT_ID) as CITY,COUNTRY,COUNTRY_CODE,update_ZIP_CODE_udf(ACCOUNT_ID) as ZIP_CODE,update_SWIFT_ADDR_udf(SWIFT_ADDR) as SWIFT_ADDR,update_TEL_NUM_udf(ACCOUNT_ID) as TEL_NUM,update_EMAIL_ADDR_udf(ACCOUNT_ID) as EMAIL_ADDR,update_CNTCT_PRSN_udf(ACCOUNT_ID) as CNTCT_PRSN,update_CMPNY_NAME_udf(ACCOUNT_ID) as CMPNY_NAME,update_FAX_NUM_udf(ACCOUNT_ID) as FAX_NUM from test_bics1 limit 100")

#Write the file to DataBase table test_bics
df1.write.mode("overwrite").jdbc(url=url, table="test_bics2", properties=properties)

val2 = df.count()
print val2


if val1 == val2:
    print "All recourds uploaded"
else:
    print "Record mismatch1"