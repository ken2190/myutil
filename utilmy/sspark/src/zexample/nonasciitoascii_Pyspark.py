#Create a Method to handle the Non Ascii to Ascii conversion
def nonasciitoascii(unicodestring):
  return unicodestring.encode("ascii","ignore")

#Create a Sample Dataframe
from pyspark.sql.window import Window
from pyspark.sql.functions import count, col
from pyspark.sql import Row
d=[ Row(coltype='regular', value="Happy Coding"),
    Row(coltype='non ascii', value="hello aåbäcö"),
    Row(coltype='non ascii',value="6Â 918Â 417Â 712"),
    Row(coltype='non ascii',value="SAN MATEOï¿½ ï¿½?A "),
    Row(coltype='non ascii',value="SAINT-LOUIS (CANADAï¿½ ï¿½ AL)")]

data = sqlContext.createDataFrame(d)
#data.show()

data = sqlContext.createDataFrame(d)

#Apply this Conversion on the Dataframe
convertedudf = udf(nonasciitoascii)
converted = data.select('coltype','value').withColumn('converted',convertedudf(data.value))
converted.show()
