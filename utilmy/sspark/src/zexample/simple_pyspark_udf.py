import pyspark.sql.functions as F
import pyspark.sql.types as T


def lower_case(x):
    """
        This UDF takes array of string as input and returns array of lowercased strings
    """
    res = []
    for x_ in x:
        res.append(x_.lower())
    return res

convert_to_lower = F.udf(lower_case, T.ArrayType(T.StringType()))