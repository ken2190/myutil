from pyspark.sql import SparkSession, SQLContext, DataFrame, GroupedData, Window, Column, Row
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, TimestampType, DoubleType, \
    BooleanType
import pyspark.sql.functions as F

# import pandas as pd

spark: SparkSession = SparkSession.builder.appName('Basics').master('local[*]').getOrCreate()
# sc = SparkContext()
# sqlContext = SQLContext(sc)

# stockSchema = StructType([StructField('Date', TimestampType(), False), StructField('Open', DoubleType(), False),
#                      StructField('High', DoubleType(), False), StructField('Low', DoubleType(), False),
#                      StructField('Close', DoubleType(), False), StructField('Volume', IntegerType(), False),
#                      StructField('Adj Close', DoubleType(), False)])

# df = spark.read.json('../data/people.json')


# inputSchema = df.schema

def getStringSchema(schema):
    stringSchema = []
    colNames = []
    for attr in schema.fields:
        stringSchema.append(StructField(attr.name, StringType(), attr.nullable, attr.metadata))
        colNames.append(F.col(attr.name))

    return (StructType(fields=stringSchema), colNames)

"""    updatedSchemaAndColNames = getStringSchema(stockSchema)
    df= spark.read.csv('../data/appl_stock.csv', header=True, schema=StructType(fields=updatedSchemaAndColNames[0]))
    # df.printSchema()
    # df.show()
    concatRowColExpr = F.concat_ws('|', *updatedSchemaAndColNames[1])
    stringdf = df.withColumn('concat_row', concatRowColExpr)"""

def getStringDF(spark, headerFlag, fileLocation, schema, prefilter):
    updatedSchemaAndColNames = getStringSchema(schema)
    df= spark.read.csv(fileLocation, header=headerFlag, schema=schema)
    # df.printSchema()
    # df.show()
    concatRowColExpr = F.concat_ws('|', *updatedSchemaAndColNames[1])
    stringdf = df.withColumn('concat_row', concatRowColExpr)
    filteredDF = stringdf
    for filtercondition in prefilter:
        filteredDF = stringdf.filter(filtercondition)
    return filteredDF

header = True
filePath = '../data/appl_stock.csv'
# df = getStringDF(spark, header, filePath, stockSchema, [])
# df.show()

from pyspark.accumulators import AccumulatorParam
class ListParam(AccumulatorParam):
    def zero(self, v):
        return []
    def addInPlace(self, variable, value):
        variable.append(value)
        return variable

exceptionCounter = spark.sparkContext.accumulator(0)
exceptionAccumulator = spark.sparkContext.accumulator([],ListParam())
exprList = []
rulesList = []

#Exception messages
dqTypeCastExceptionString = "DQ Issue found with {} --> {}. \nField incompatible with defined type: {}, Refer record \n {}"
dqNullableExceptionString = "DQ Issue found with {} --> {}. \nField incompatible with defined type: {}, Refer record \n {}"

castExpr = "cast({} as {})"

def getNullableStringTypeSqlExpr(field, failEarlyFlag):
    sqlExpr = "when {} is null then throwNullableStringException('{}', {}, '{}', concat_row, '{}') ".format(
        field.name, field.name, field.name, field.dataType, failEarlyFlag, field.name, field.name)
    return sqlExpr

def getNullableBooleanTypeSqlExpr(field, failEarlyFlag):
    sqlExpr = "when {} is null then throwNullableBooleanException('{}', {}, '{}', concat_row, '{}') ".format(
        field.name, field.name, field.name, field.dataType, failEarlyFlag, field.name, field.name)
    return sqlExpr

def getStringTypeSqlExpr(field, failEarlyFlag, nullableExpr, sqlExpr):
    sqlExpr += " case when {} is null then throwNullableStringException('{}', {}, '{}', concat_row, '{}') else {} end as {}".format(
        field.name, field.name, field.name, field.dataType, failEarlyFlag, field.name, field.name)
    return sqlExpr

def getBinaryTypeSqlExpr(field, failEarlyFlag, nullableExpr, sqlExpr):
    castExpr2 = castExpr.format(field.name, field.dataType.jsonValue())
    sqlExpr += " case {} when {} is not null and {} is null then throwBinaryTypeCastException('{}', {}, '{}', concat_row, '{}') else {} end as {}".format(
        nullableExpr, field.name, castExpr2, field.name, field.name, field.dataType, failEarlyFlag, castExpr2, field.name)
    return sqlExpr

def getBooleanTypeSqlExpr(field: StructField, failEarlyFlag, nullableExpr, sqlExpr):
    castExpr2 = castExpr.format(field.name, field.dataType.jsonValue())
    sqlExpr += " case {} when {} is not null and {} is null then throwBooleanTypeCastException('{}', {}, '{}', concat_row, '{}') else {} end as {}".format(
        nullableExpr, field.name, castExpr2, field.name, field.name, field.dataType, failEarlyFlag, castExpr2, field.name)
    return sqlExpr


def getTypeCastBooleanException(counter, accumulator, attr, attrVal, attrType, record, failEarlyFlag):
    counter+=1
    accumulator.add(dqTypeCastExceptionString.format(attr, attrVal, attrType, record))
    if failEarlyFlag.lower() == 'true':
        raise Exception(dqTypeCastExceptionString.format(attr, attrVal, attrType, record))
    else:
        return dqTypeCastExceptionString.format(attr, attrVal, attrType, record)

def getNullableStringException(counter, accumulator, attr, attrVal, attrType, record, failEarlyFlag):
    counter += 1
    accumulator.add(dqNullableExceptionString.format(attr, attrVal, attrType, record))
    if failEarlyFlag.lower() == 'true':
        raise Exception(dqTypeCastExceptionString.format(attr, attrVal, attrType, record))
    else:
        return dqNullableExceptionString.format(attr, attrVal, attrType, record)
 
# typeCastBoolException = F.udf(lambda attr, attrVal, attrType, record, failEarlyFlag: getTypeCastBooleanException(exceptionCounter, exceptionAccumulator, attr, attrVal, attrType, record, failEarlyFlag), BooleanType())
typeCastBoolException = F.udf(lambda attr, attrVal, attrType, record, failEarlyFlag: getTypeCastBooleanException(exceptionCounter, exceptionAccumulator, attr, attrVal, attrType, record, failEarlyFlag), BooleanType())
nullableStringException = F.udf(lambda attr, attrVal, attrType, record, failEarlyFlag: getNullableStringException(exceptionCounter, exceptionAccumulator, attr, attrVal, attrType, record, failEarlyFlag), StringType())
nullableBooleanException = F.udf(lambda attr, attrVal, attrType, record, failEarlyFlag: getNullableStringException(exceptionCounter, exceptionAccumulator, attr, attrVal, attrType, record, failEarlyFlag), BooleanType())

spark.udf.register("throwBooleanTypeCastException", typeCastBoolException)
spark.udf.register("throwNullableStringException", nullableStringException)
spark.udf.register("throwNullableBooleanException", nullableBooleanException)

dataTypeSqlExprFn = {
    "StringType" : getStringTypeSqlExpr,
    "BooleanType" : getBooleanTypeSqlExpr
}

nullableSqlExprFn = {
    "StringType" : getNullableStringTypeSqlExpr,
    "BooleanType" : getNullableBooleanTypeSqlExpr
}

def getSqlExpr(field: StructField, failEarlyFlag):
    sqlExpr = ""
    attrName = field.name
    attrType = field.dataType
    print(str(attrType))
    nullableExpr = ""
    if not field.nullable:
        # print(field.name+" is nullable: " + str(field.nullable))
        nullableExpr = nullableSqlExprFn.get(str(attrType))(field, failEarlyFlag)

    sqlExpr = dataTypeSqlExprFn.get(str(attrType))(field, failEarlyFlag, nullableExpr, sqlExpr)
    print(sqlExpr)
    return sqlExpr


stockSchema = StructType(fields=[StructField('name', StringType(), True), StructField('age', BooleanType(), False)])
# stockSchema = StructType(fields=[StructField('name', StringType(), True)])

updatedSchemaAndColNames = getStringSchema(stockSchema)
concatRowColExpr = F.concat_ws('|', *updatedSchemaAndColNames[1])

df = spark.read.json('../data/people.json', schema=updatedSchemaAndColNames[0]).withColumn('concat_row', concatRowColExpr)
df.show()
for attr in stockSchema.fields:
    exprList.append((attr.name, getSqlExpr(attr, False)))

for x in exprList:
    print(x[0], x[1])
    df = df.withColumn(x[0]+"_rule", F.expr(x[1]))
# df.selectExpr(*exprList).show()

df.show()

print(exceptionAccumulator.value)
print(exceptionCounter.value)