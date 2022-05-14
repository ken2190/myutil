import pyspark.sql.functions as F
import pyspark.sql.types as T
import json

# UDF to parse array stored as string using JSON
def parse_array_from_string(x):
	res = json.loads(x)
	return res

retrieve_array = F.udf(parse_array_from_string, T.ArrayType(T.StringType()))