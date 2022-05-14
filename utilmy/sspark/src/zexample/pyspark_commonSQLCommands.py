'''
1. Loading XML's into pyspark dataframes
'''
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages com.databricks:spark-xml_2.10:0.4.1 pyspark-shell'
import findspark
findspark.init('/location/of/spark2-client/')
import pyspark
from pyspark.sql import SQLContext,SparkSession, HiveContext
spark = SparkSession.builder.appName('NAME_OF_JOBS').enableHiveSupport().getOrCreate()

xml_filenames = '/location/of/xmls/in/hdfs/*.xml'
df = spark.read.format('com.databricks.spark.xml').options(rowTag='Tag_Staring_Each_Row_eg: exch:exchange-document').load(xml_filenames)

'''
2. Count the Unique or Distinct cales in a column
'''
from pyspark.sql.functions import *
df.agg(countDistinct(col('COLUMN_NAME')).alias('NEW_FIELD_NAME')).collect()[0][0]

'''
3. Create a Pyspark UDF With Two 2 Columns as Inputs. returns an Array of values for New Column
'''
def compare_two_columns(struct_cols):
    col_1 = struct_cols[0]
    col_2 = struct_cols[1]
    return_array = []
    for item_A in col_1:
        for item_B in col_2:
            if condition:
                result = 'Compute Something'
                return_array.append(result) 
    return return array
compare_two_columns_udf = udf(compare_two_columns, ArrayType(StringType()))

df = df.withColumn('newColumn', compare_two_columns_udf(
  struct(firstCol.cast(ArrayType(StringType())), secondCol.cast(ArrayType(StringType()))))
                   
'''
4. Set Print Out Log Level
'''
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Image_Vectorization').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
                   
'''
5. Creating Small DF's
'''

#This gives an error: 'Can Not Infer Schema'
item_list =['x','y','z'] 
df = spark.createDataFrame(item_list, ['text'])
df.show()

#If you want to put an array in a column:
items =['x','y','z'] 
lines = ([items],)
df = spark.createDataFrame(lines, ['text'])
df.show()

#If you want to make each item its own row:
items =[['x'],['y'],['z']] 
df = spark.createDataFrame(items, ['text'])
df.show()
                   
'''
6. Converting a Column of Array to Vector Type
https://stackoverflow.com/questions/42138482/pyspark-how-do-i-convert-an-array-i-e-list-column-to-vector
'''
#Assuming df is of two columns: 1='ID', StringType(), 2='list', ArrayType(DoubleType())
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df_vec = df.select('ID',list_to_vector_udf('List').alias('list_VectType'))       
                   
'''
7. Granting Privledges
'''
GRANT ALL ON pq_ba_local.pq_bi_testing TO productqualitydb_rs_pq_bi_ro
GRANT ALL PRIVILEGES ON TABLE productqualitydb.pq_ba_local.pq_bi_asins TO productqualitydb_rs_pq_bi_ro

               
                                                                      