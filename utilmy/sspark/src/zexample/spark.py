# Row, Column, DataFrame, value are different concepts, and operating over DataFrames requires
# understanding these differences well.
#
# withColumn + UDF | must receive Column objects in the udf
# select + UDF | udf behaves as a mapping




from pyspark.sql import SparkSession
from pyspark.sql.types import (StructField
                               ,StringType
                               , IntegerType
                               , StructType)

# Spark infering schema ######################################################################
> df = spark.read.json('people.json')
> df.printSchema()
root
 |-- age: long (nullable = true)
 |-- name: string (nullable = true)

# Defining schema independt of Spark infering
> data_schema = [StructField('age', IntegerType(), True)
               ,StructField('name', StringType(), True)]
> final_struct = StructType(fields = data_schema) 
> df = spark.read.json('people.json', schema=final_struct)
> df.printSchema()
root
 |-- age: integer (nullable = true)
 |-- name: string (nullable = true)
  
# Column / Row vs. DataFrames #####################################################################
# DF are more flexible than columns
> type(df['age'])
pyspark.sql.column.Column

> type(df.select('age'))
pyspark.sql.dataframe.DataFrame
> df.select(['age', 'name']).show()
+----+-------+
| age|   name|
+----+-------+
|null|Michael|
|  30|   Andy|
|  19| Justin|
+----+-------+

> df.withColumn('new_age', df['age'] * 2)

# Rows : arrays of Row objects
> type(df.head(2)[0])
pyspark.sql.types.Row
> df.head(2)
[Row(age=None, name='Michael'), Row(age=30, name='Andy')]

> df.show()
+-------------------+------------------+------------------+------------------+------------------+--------+------------------+
|               Date|              Open|              High|               Low|             Close|  Volume|         Adj Close|
+-------------------+------------------+------------------+------------------+------------------+--------+------------------+
|2012-01-03 00:00:00|         59.970001|         61.060001|         59.869999|         60.330002|12668800|52.619234999999996|
...

> df.select(df['High'].desc()).head(1)[0]
Row(
  Date=datetime.datetime(2015, 1, 13, 0, 0)
  , Open=90.800003
  , High=90.970001
  , Low=88.93
  , Close=89.309998
  , Volume=8215400
  , Adj Close=83.825448
)
> df.select(df['High'].desc()).head(1)[0].Date.day
13


# SQL QUERIES ######################################################################
> df.createOrReplaceTempView('people')
> results = spark.sql("select * from people where age < 30")
> results.show()
+---+------+
|age|  name|
+---+------+
| 19|Justin|
+---+------+

# FILTERING #########################################################################
> from pyspark.sql import SparkSession
> spark = SparkSession.builder.appName('ops').getOrCreate()
> df = spark.read.csv('appl_stock.csv', inferSchema=True, header=True)
> df.head(1)[0]
Row(Date=datetime.datetime(2010, 1, 4, 0, 0), Open=213.429998, High=214.499996, Low=212.38000099999996, Close=214.009998, Volume=123432400, Adj Close=27.727039)

> # df.filter("Close < 500").select(['Open', 'Close']).show() # with SQL syntax
> df.filter((df['Close'] < 500) & ~(df['Open'] > 200)).select(['Open', 'Close']).show()
+------------------+------------------+
|              Open|             Close|
+------------------+------------------+
|192.36999699999998|        194.729998|
|        195.909998|        195.859997|
...
|         91.510002|         92.199997|
|         92.309998| 92.08000200000001|
+------------------+------------------+

> res = df.filter(df['Low'] == 197.16).collect()
> res[0].asDict()
{'Adj Close': 25.620401,
 'Close': 197.75,
 'Date': datetime.datetime(2010, 1, 22, 0, 0),
 'High': 207.499996,
 'Low': 197.16,
 'Open': 206.78000600000001,
 'Volume': 220441900}

# AGGREGATIONS #########################################################################
> spark = SparkSession.builder.appName('aggs').getOrCreate()
> df = spark.read.csv('sales_info.csv', inferSchema=True, header=True)
> df.printSchema()
> df.groupBy('Company').sum().show()
+-------+----------+
|Company|sum(Sales)|
+-------+----------+
|   APPL|    1480.0|
|   GOOG|     660.0|
|     FB|    1220.0|
|   MSFT|     967.0|
+-------+----------+
> df.groupBy('Company').agg({'Sales' : 'sum'}).show()
+-------+----------+
|Company|sum(Sales)|
+-------+----------+
|   APPL|    1480.0|
|   GOOG|     660.0|
|     FB|    1220.0|
|   MSFT|     967.0|
+-------+----------+

> df.agg({'Sales': 'sum'}).show()
+----------+
|sum(Sales)|
+----------+
|    4327.0|
+----------+

> df.agg({'Sales': 'max'}).show()
+----------+
|max(Sales)|
+----------+
|     870.0|
+----------+

> df2.select('High').show()
+------------------+
|              High|
+------------------+
|         61.060001|
...
|             61.57|
+------------------+

> high_peak = df2.agg({'High': 'max'}).collect()[0]
> high_peak = high_peak['max(High)']
> high_peak
90.970001

> from pyspark.sql.functions import countDistinct
> days_total = file.agg(countDistinct('Date').alias('cnt')).collect()[0].cnt
1258


# https://stackoverflow.com/questions/40888946/spark-dataframe-count-distinct-values-of-every-column/40889920
> df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns))

> file.agg({'Close': 'mean'}).withColumnRenamed('avg(Close)', 'AVG').select(format_number('AVG', 2)).show()
+---------------------+
|format_number(AVG, 2)|
+---------------------+
|                72.39|
+---------------------+


## aggregating on DATE column and on STRING column (parseable to date)
>>> df.groupBy('position').agg(F.max('end')).show()
+--------+----------+
|position|  max(end)|
+--------+----------+
|   POS_1|2002-01-01|
+--------+----------+

>>> df.select('position', F.col('end').cast('string')).groupBy('position').agg(F.max('end')).show()
+--------+----------+
|position|  max(end)|
+--------+----------+
|   POS_1|2002-01-01|
+--------+----------+


# SPEC FUNCTIONS ################################################################
> from pyspark.sql.functions import countDistinct, avg, stddev, format_number
> df.select(stddev('Sales').alias('std'))\
  .select(format_number('std', 2).alias('std')).show()
+------+
|   std|
+------+
|250.09|
+------+

> from pyspark.sql.functions import max, min
> df.select(max('Volume').alias('MAX'), min('Volume').alias('MIN')).show()
+--------+-------+
|     MAX|    MIN|
+--------+-------+
|80898100|2094900|
+--------+-------+

> df[df['Close'] < 60].count()
> df.filter('Close < 60').count()
> df.filter(df['Close'] < 60).count()
81

> from pyspark.sql.functions import count
> file.filter(file['Close'] < 60).select(count('Close').alias('COUNT')).show()
+-----+
|COUNT|
+-----+
|   81|
+-----+

> from pyspark.sql.functions import corr
> file.select(corr('High', 'Volume').alias('Pearson Correlation')).show()
+-------------------+
|Pearson Correlation|
+-------------------+
|-0.3384326061737161|
+-------------------+


# user defined functions | UDF #####################################################################
# udfs are applied to col elements, not to cols
from pyspark.sql import functions as F

>>> def f(c1, c2):
      return str(c1) + str(c2)
>>> fu = F.udf(f, StringType())
>>> df = spark.createDataFrame([(1, 'a'), (1, 'b'), (2, 'd')], ['c1', 'c2'])
>>> df.withColumn('test', fu(df.c1, df.c2)).show()
+---+---+----+
| c1| c2|test|
+---+---+----+
|  1|  a|  1a|
|  1|  b|  1b|
|  2|  d|  2d|
+---+---+----+




# ORDERING #####################################################################
> df.orderBy(df['Sales'].desc()).show() # for desc
> df.orderBy('Sales').show()

# grouping by MONTH, getting the avg of 'Close' and displaying by DESC order, to 2 signf digits
> from pyspark.sql.functions import month, mean, col
> (file 
 .withColumn('MONTH', month(file['Date']).alias('MONTH'))
 .groupBy('MONTH')
 .mean()
 .select('MONTH', 'avg(Close)')
 .sort(col('MONTH').desc()) # or .orderBy('Month')
 .select('Month', format_number('avg(Close)', 2))
 .show())

# CLEANING ####################################################################
> spark = SparkSession.builder.appName('miss').getOrCreate()
> df = spark.read.csv('ContainsNull.csv', header = True, inferSchema=True)
> df.show()
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp1| John| null|
|emp2| null| null|
|emp3| null|345.0|
|emp4|Cindy|456.0|
+----+-----+-----+

> df.na.drop(thresh = 2).show() # at least N non-null values, 0 by default
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp1| John| null|
|emp3| null|345.0|
|emp4|Cindy|456.0|
+----+-----+-----+

> df.na.drop(how = 'any').show() # (any | all) null?
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp4|Cindy|456.0|
+----+-----+-----+

> df.na.drop(how = 'any', subset=['Sales']).show() # on the columns specified
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp3| null|345.0|
|emp4|Cindy|456.0|
+----+-----+-----+

> # df.na.fill(0).show() # it won't fill string cols
> df.na.fill(0, subset=['Sales']).show()
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp1| John|  0.0|
|emp2| null|  0.0|
|emp3| null|345.0|
|emp4|Cindy|456.0|
+----+-----+-----+

> from pyspark.sql.functions import mean
> mean_val = df.select(mean(df['Sales'])).collect()
> mean_val = mean_val[0][0]
> df.na.fill(mean_val, ['Sales']).show() # not nec to spec arg
+----+-----+-----+
|  Id| Name|Sales|
+----+-----+-----+
|emp1| John|400.5|
|emp2| null|400.5|
|emp3| null|345.0|
|emp4|Cindy|456.0|
+----+-----+-----+


# DATES #####################################################################################
> spark = SparkSession.builder.appName('dates').getOrCreate()
> df = spark.read.csv('appl_stock.csv', header = True, inferSchema = True)
> from pyspark.sql.functions import (dayofmonth, hour, dayofyear, month, year, 
                                   weekofyear, format_number, date_format)
> df.head(1)[0]
Row(Date=datetime.datetime(2010, 1, 4, 0, 0)
    , Open=213.429998, High=214.499996
    , Low=212.38000099999996, Close=214.009998
    , Volume=123432400, Adj Close=27.727039)

> new_df = df.withColumn('Year', year(df['Date']))
> res1 = new_df.groupBy('Year').mean().select(['Year', 'avg(Close)']) # applying the 'mean' function to every column
> res2 = res1.withColumnRenamed('avg(Close)', 'Avg Closing Price')
> res3 = res2.select(['Year', format_number('Avg Closing Price', 2).alias('AVG')]).show()
+----+------+
|Year|   AVG|
+----+------+
|2015|120.04|
|2013|472.63|
|2014|295.40|
|2012|576.05|
|2016|104.60|
|2010|259.84|
|2011|364.00|
+----+------+

> from pyspark.sql.functions import min, max, to_timestamp
> nooa_df.select(min('DATE').alias('min'), max('DATE').alias('max')).select(to_timestamp('min', 'yyyy-MM-dd').alias('min'), to_timestamp('max', 'yyyy-MM-dd').alias('max')).show()
+-------------------+-------------------+
|                min|                max|
+-------------------+-------------------+
|2009-01-01 00:00:00|2017-12-31 00:00:00|
+-------------------+-------------------+

# DATE from DATETIME
> green_2013_grouped_df = green_2013_filtered_df.withColumn('date', to_date('datet')).groupBy('date').count()
+----------+-----+
|      date|count|
+----------+-----+
|2013-09-09|  942|
...

# Verify group by with filter
> from pyspark.sql.functions import lit, to_date
> green_2013_filtered_df.filter(to_date(green_2013_filtered_df['datet']) == lit("2013-09-09")).count()
942

# GROUP BY ####################################################################################
> file.withColumn('YEAR', year(file['Date'])).groupBy('YEAR').max().show()
+----+-----------------+---------+---------+----------+-----------+-----------------+---------+
|YEAR|        max(Open)|max(High)| max(Low)|max(Close)|max(Volume)|   max(Adj Close)|max(YEAR)|
+----+-----------------+---------+---------+----------+-----------+-----------------+---------+
|2015|        90.800003|90.970001|    89.25| 90.470001|   80898100|84.91421600000001|     2015|
...
|2016|             74.5|75.190002|73.629997| 74.300003|   35076700|        73.233524|     2016|
+----+-----------------+---------+---------+----------+-----------+-----------------+---------+

# the groupby clause was attained by generating a column first, the year derived from the date
> file.withColumn('YEAR', year(file['Date'])).groupBy('YEAR').max().select('YEAR', 'max(High)').show()
+----+---------+
|YEAR|max(High)|
+----+---------+
|2015|90.970001|
...
|2016|75.190002|
+----+---------+

> df = spark.createDataFrame([('d1', 'r1'), ('d1', 'r2'), ('d1', 'r3'), ('d2', 'r4'), ('d2', 'r5')], ['date','ride'])
> df.groupBy('date').count().show()
+----+-----+
|date|count|
+----+-----+
|  d2|    2|
|  d1|    3|
+----+-----+

# MAP, FILTER, REDUCE #################################################################################################
> attributes = (
      nooa_df.select(field_attr)
      .distinct()
      .orderBy(nooa_df[field_attr].asc())
      .rdd.map(lambda _: _[field_attr])
      .collect()
  )

> df_2 = spark.createDataFrame([
        ('d1', 'r1')], ['date','ride'])
> df_3 = spark.createDataFrame([
        ('d3', 'r1')
        ], ['date','ride'])
> reduce((lambda x, y: x.union(y)), [df_2, df_3]).show()
+----+----+
|date|ride|
+----+----+
|  d1|  r1|
|  d3|  r1|
+----+----+

## CAREFUL! ##########

>>> df_3.show()
+----+----+----+
|ride|date|fare|
+----+----+----+
|  r1|  d3|    |
+----+----+----+

>>> df_4.show()
+----+----+----+
|date|ride|fare|
+----+----+----+
|  d3|  r1|  12|
+----+----+----+

>>> reduce((lambda x, y: x.union(y)), [df_4, df_3]).show()
+----+----+----+
|date|ride|fare|
+----+----+----+
|  d3|  r1|  12|
|  r1|  d3|    |
+----+----+----+

>>> reduce((lambda x, y: x.union(y)), [df_4.select(df_4.columns), df_3.select(df_4.columns)]).show()
+----+----+----+
|date|ride|fare|
+----+----+----+
|  d3|  r1|  12|
|  d3|  r1|    |
+----+----+----+

## MAP ##
# mapping occurs with a withColumn, see UDF
>>> @F.udf(returnType=T.StringType())
    def f(c1, c2):
      return str(c1) + str(c2)
>>> df = spark.createDataFrame([(1, '123'), (1, '90'), (2, '45')], ['c1', 'c2'])
>>> df.withColumn('test', f(df.c1, df.c2)).show()
+---+---+----+
| c1| c2|test|
+---+---+----+
|  1|123|1123|
|  1| 90| 190|
|  2| 45| 245|
+---+---+----+




## FILTER ##

>>> @F.udf(T.BooleanType())
    def g(c1, c2):
    	return int(c1) > 1 & int(c2) % 2 == 0
>>> df = spark.createDataFrame([(1, '123'), (1, '90'), (2, '45')], ['c1', 'c2'])
>>> df.filter(g(df.c1, df.c2)).show()
+---+---+
| c1| c2|
+---+---+
|  1| 90|
+---+---+


# JOIN ###############################################################################################################

# INNER
# Eliminate non matching rows from both dataframes

> df_1 = spark.createDataFrame([
        ('p1', 'd1')
        , ('p2', 'd2')], ['precipitation','date'])
> df_2 = spark.createDataFrame([
        ('d1', 'r1')
        , ('d3', 'r2')], ['date','ride'])
> df_1.join(df_2, 'date', 'inner').show()
+----+-------------+----+
|date|precipitation|ride|
+----+-------------+----+
|  d1|           p1|  r1|
+----+-------------+----+


# LEFT / RIGHT 
# LEFT | eliminate non-matching rows from the 2nd dataframe (on the right)
# RIGHT | eliminate non-matching rows from the 1st dataframe (on the left)

> df_1.join(df_2, 'date', 'left').show()
+----+-------------+----+
|date|precipitation|ride|
+----+-------------+----+
|  d2|           p2|null|
|  d1|           p1|  r1|
+----+-------------+----+
> df_2.join(df_1, 'date', 'right').show()
+----+----+-------------+
|date|ride|precipitation|
+----+----+-------------+
|  d2|null|           p2|
|  d1|  r1|           p1|
+----+----+-------------+

# OUTER
# Get all rows from both dataframes

> df_1.join(df_2, 'date', 'outer').show()
> df_2.join(df_1, 'date', 'outer').show()
+----+-------------+----+
|date|precipitation|ride|
+----+-------------+----+
|  d2|           p2|null|
|  d3|         null|  r2|
|  d1|           p1|  r1|
+----+-------------+----+

+----+----+-------------+
|date|ride|precipitation|
+----+----+-------------+
|  d2|null|           p2|
|  d3|  r2|         null|
|  d1|  r1|           p1|
+----+----+-------------+

# LEFTSEMI
# useful for filtering rows with column values in distinct values of other dataframe column values

>>> df_1 = spark.createDataFrame([(1, 'a'), (2, 'a'), (3, 'b')], ['c1', 'c2'])
>>> df_2 = spark.createDataFrame([(1345, 'a'), (25345, 'a'), (345, 'a')], ['c3', 'c2'])
>>> df_1.join(df_2, df_1['c2'] == df_2['c2'], 'leftsemi').show()
+---+---+
| c1| c2|
+---+---+
|  1|  a|
|  2|  a|
+---+---+

>>> gf = spark.createDataFrame([(1, 'p1'), (3, 'p2'), (4, 'p2')], ['msn', 'position'])
+---+--------+
|msn|position|
+---+--------+
|  1|      p1|
|  3|      p2|
|  4|      p2|
+---+--------+


>>> df = spark.createDataFrame([(1, 'p1', 'o1'), (1, 'p2', 'o2'), (3, 'p2', 'o3')], ['msn', 'position', 'other'])
+---+--------+-----+
|msn|position|other|
+---+--------+-----+
|  1|      p1|   o1|
|  1|      p2|   o2|
|  3|      p2|   o3|
+---+--------+-----+


>>> df.join(gf, 'msn', 'inner').show()
+---+--------+-----+--------+
|msn|position|other|position|
+---+--------+-----+--------+
|  1|      p2|   o2|      p1|
|  1|      p1|   o1|      p1|
|  3|      p2|   o3|      p2|
+---+--------+-----+--------+

>>> df.join(gf, df['msn'] != gf['msn'], 'inner').show()
+---+--------+-----+---+--------+
|msn|position|other|msn|position|
+---+--------+-----+---+--------+
|  1|      p1|   o1|  3|      p2|
|  1|      p1|   o1|  4|      p2|
|  1|      p2|   o2|  3|      p2|
|  1|      p2|   o2|  4|      p2|
|  3|      p2|   o3|  1|      p1|
|  3|      p2|   o3|  4|      p2|
+---+--------+-----+---+--------+



>>> df.join(gf, ['msn', 'position'], 'inner').show()
+---+--------+-----+
|msn|position|other|
+---+--------+-----+
|  1|      p1|   o1|
|  3|      p2|   o3|
+---+--------+-----+


# CREATE ###############################################################################################################
# https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html

# DF FROM ARRAY
>>> l = [('Alice', 1)]
>>> spark.createDataFrame(l).collect()
[Row(_1=u'Alice', _2=1)]
>>> spark.createDataFrame(l, ['name', 'age']).collect()
[Row(name=u'Alice', age=1)]

# DF FROM RDD FROM ARRAY
>>> sc = spark.sparkContext
>>> rdd = sc.parallelize(l)
>>> spark.createDataFrame(rdd).collect()
[Row(_1=u'Alice', _2=1)]
>>> df = spark.createDataFrame(rdd, ['name', 'age'])
>>> df.collect()
[Row(name=u'Alice', age=1)]

# DF FROM (RDD + DATA_TYPES SPEC) FROM ARRAY
>>> spark.createDataFrame(rdd, "a: string, b: int").collect()
[Row(a=u'Alice', b=1)]
>>> rdd = rdd.map(lambda row: row[1])
>>> spark.createDataFrame(rdd, "int").collect()
[Row(value=1)]
>>> spark.createDataFrame(rdd, "boolean").collect() 
Traceback (most recent call last):
    ...
Py4JJavaError: ...
  
# DF FROM ROM FROM RDD FROM ARRAY
>>> from pyspark.sql import Row
>>> Person = Row('name', 'age')
>>> person = rdd.map(lambda r: Person(*r))
>>> df2 = spark.createDataFrame(person)
>>> df2.collect()
[Row(name=u'Alice', age=1)]

# DF FROM (STRUCT + RDD)
>>> from pyspark.sql.types import *
>>> schema = StructType([
...    StructField("name", StringType(), True),
...    StructField("age", IntegerType(), True)])
>>> df3 = spark.createDataFrame(rdd, schema)
>>> df3.collect()
[Row(name=u'Alice', age=1)]


# CAST ######################################################################################
>>> sc = StructType(
  [ StructField('Position', T.StringType(), True), StructField('End', T.DateType(), True) ]
)
>>> data = [ 
  ('POS_1', datetime.datetime.strptime('2000-01-01', '%Y-%m-%d').date())
  , ('POS_1', datetime.datetime.strptime('2001-01-01', '%Y-%m-%d').date())
  , ('POS_1', datetime.datetime.strptime('2002-01-01', '%Y-%m-%d').date()) 
]
>>> scont = spark.sparkContext
>>> rdd = scont.parallelize(data)
>>> df = spark.createDataFrame(rdd, ['position', 'end'])
>>> df.show()
+--------+----------+
|position|       end|
+--------+----------+
|   POS_1|2016-01-01|
|   POS_1|2017-01-01|
|   POS_1|2015-01-01|
+--------+----------+
>>> df.describe
<bound method DataFrame.describe of DataFrame[position: string, end: date]>
>>> df.select('position', F.col('end').cast('string')).describe
<bound method DataFrame.describe of DataFrame[position: string, end: string]>


