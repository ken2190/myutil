#by default, you can't apply directly a regular python function to an entire Column of a dataframe
#for that you need to define a UDF function, with the following :

from pyspark.sql.functions import udf

# you first define your function normally
def transfo(df.col):
    return (df.col + 3)
  
# you then declare a udf function that is basically a lamdba applying your first regular function
udf_transfo = udf(lambda x : transfo(x))


def transfo(df.col):
    return json.loads(df.col)

#if the output type of your first function is different than the input type, you need to explicitly define the output type in the udf declaration
# import all pypsark types first
from pyspark.sql.types import *
# declare udf with specific output type (here a mapType with keys as strings and values as integers)
udf_transfo = udf(lambda x : transfo(x), MapType(StringType(),IntegerType()))