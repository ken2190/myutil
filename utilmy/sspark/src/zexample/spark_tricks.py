from pyspark.sql.functions import countDistinct, avg, stddev
from pyspark.sql.functions import format_number
# aggregations
df.select(countDistinct("Sales")).show()
df.select(avg("Sales").alias("avgSales")).show()
df.orderBy("Sales").show()
df.orderBy("Company").show()
df.orderBy(df["Sales"].desc()).show()

sales_std = df.select(stddev("Sales").alias("Sales Std"))
sales_std.select(format_number("Sales Std",2).alias("Sales Std")).show()

# datetime 
from pyspark.sql.functions import dayofmonth,dayofyear,weekofyear,date_format
from pyspark.sql.functions import month,year
from pyspark.sql.functions import hour,minute,format_number
df.select(dayofmonth(df["Date"])).show()
df.select(year(df["Date"])).show()

# Row format
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *
df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()

df = df.select("medianHouseValue", "totalBedRooms", "population") 
df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households"))
df = df.withColumn("medianHouseValue",  col("medianHouseValue")/100000)
df = df.withColumn( "longitude", df["longitude"].cast(FloatType()) ) 
       .withColumn( "latitude",  df["latitude"].cast(FloatType())  ) 
df.select(col("population")/col("households"))
df.select('population','totalBedRooms').show(10)
df.describe().show()

# aggregations
df.groupBy("housingMedianAge").count().sort("housingMedianAge",ascending=False).show()
# udf functions
def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 

columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 
           'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']
df = convertColumn(df, columns, FloatType())


# udf functions
from pyspark.sql.functions import *
get_domain = udf(lambda x: re.search("@([^@]*)", x = "@").group(1))
df.select(get_domain(df.commiteremail).alias("domain"))
  .groupBy("domain").count()
  .orderBy(desc("count")).take(5)

# efficient joins
myUDF = udf(lambda x,y: x == y)
df1.join(df2, myUDF(col("x"), col("y")) )