from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import udf


def tokenize(seq):
    return tokenizer(seq)['input_ids']


tokenize_udf = udf(tokenize, ArrayType(IntegerType()))