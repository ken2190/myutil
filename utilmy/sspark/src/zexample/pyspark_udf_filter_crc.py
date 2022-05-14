from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
import zlib

def get_pct(uid, mod=100):
    return zlib.crc32(uid) % mod 

def pct_filter(uid, pct = 0):    
    return get_pct(uid) == pct

pct_filter_udf = udf(pct_filter, BooleanType())

df = df.filter(pct_filter_udf(df.uid))