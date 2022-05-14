"""
PySpark tokenizer transformer

PySpark transformer that supports format-preserving encryption using the FFX1 algorithm (from pyffx library)

requirements:
  pyffx
  pyspark
"""
from string import digits, ascii_uppercase, ascii_lowercase

import pyffx
from pyspark.sql import types as T
from pyspark.sql import functions as F


def tokenize_numeric(value, key):
    if value is None or value == 'null':
        return None
    enc = pyffx.Integer(key.encode(), length=len(str(value)))
    return enc.encrypt(value)


def tokenize_string(plain, key):
    if plain is None or plain == 'null':
        return None
    plain = plain.strip()
    if plain == '':
        return None
    str_val = ''.join([c for i, c in enumerate(plain) if c.isalpha()])
    str_idx = [i for i, c in enumerate(plain) if c.isalpha()]
    num_val = ''.join([c for i, c in enumerate(plain) if c.isdigit()])
    num_idx = [i for i, c in enumerate(plain) if c.isdigit()]
    sym_val = ''.join([c for i, c in enumerate(plain) if not c.isalnum()])
    sym_idx = [i for i, c in enumerate(plain) if not c.isalnum()]
    str_enc = pyffx.String((key + plain).encode(), alphabet=ascii_lowercase + ascii_uppercase, length=len(str_val))
    num_enc = pyffx.String((key + plain).encode(), alphabet=digits, length=len(num_val))
    str_tok = str_enc.encrypt(str_val) if len(str_val) > 0 else ''
    num_tok = num_enc.encrypt(num_val) if len(num_val) > 0 else ''
    tok_list = [None] * (len(str_idx) + len(num_idx) + len(sym_idx))
    for key_index, tok_index in enumerate(str_idx):
        tok_list[tok_index] = str_tok[key_index]
    for key_index, tok_index in enumerate(num_idx):
        tok_list[tok_index] = num_tok[key_index]
    for key_index, tok_index in enumerate(sym_idx):
        tok_list[tok_index] = sym_val[key_index]
    token = ''.join(tok_list)
    return token


tokenize_string_udf = F.udf(tokenize_string, T.StringType())
tokenize_integer_udf = F.udf(tokenize_numeric, T.IntegerType())
tokenize_long_udf = F.udf(tokenize_numeric, T.LongType())


class Tokenizer(object):
    def __init__(self, key):
        self.key = key

    def transform(self, df, schema):
        for field in schema:
            treatment = field.metadata.get('treatment')
            if treatment == 'tokenize':
                df = self.tokenize_field(df, field)
        return df

    def tokenize_field(self, df, field):
        if field.dataType == T.IntegerType():
            df = df.withColumn(field.name, tokenize_integer_udf(F.col(field.name), F.lit(self.key)))
        elif field.dataType == T.LongType():
            df = df.withColumn(field.name, tokenize_long_udf(F.col(field.name), F.lit(self.key)))
        else:
            df = df.withColumn(field.name, tokenize_string_udf(F.col(field.name), F.lit(self.key)))
        return df
