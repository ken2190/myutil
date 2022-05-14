# Launch pyspark using below command
# pyspark --master yarn --conf spark.ui.port=12901

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StringType

streamingDataFrame = spark. \
  readStream. \
  format("text"). \
  load("s3n://itversitydata/2018/11/28/*")

streamingDataFrame.printSchema()

def getdate(path):
    return "-".join(path.split('/')[3:6])


def gethr(path):
    return int(path.split('/')[6])

getdate_udf=udf(getdate, StringType())
gethr_udf=udf(gethr, IntegerType())


streamingDataFrame = streamingDataFrame. \
  filter(split(split(streamingDataFrame['value'], ' ')[6], '/')[1] == 'department'). \
  withColumn('filePath', input_file_name().cast('string'))

streamingDataFrame = streamingDataFrame. \
                      withColumn('department_name', split(split(streamingDataFrame['value'], ' ')[6], '/')[2]). \
                      withColumn('dt', getdate_udf(streamingDataFrame['filePath']).cast('date')). \
                      withColumn('hr', gethr_udf(streamingDataFrame['filePath']))

streamingDataFrame = streamingDataFrame.drop('filePath')

streamingDataFrame.printSchema()

dataStreamWriter = streamingDataFrame.writeStream.format("parquet"). \
    format('console'). \
    outputMode('append'). \
    trigger(processingTime='2 seconds')

streamingQuery = dataStreamWriter.start()
    
dataStreamWriter = streamingDataFrame.writeStream.format("parquet"). \
    option("checkpointLocation", "s3n://itversitydata/checkpoint/"). \
    outputMode('append'). \
    trigger(processingTime='15 seconds'). \
    partitionBy('dt','hr')

streamingQuery = dataStreamWriter.start("s3n://itversitydata/output/")