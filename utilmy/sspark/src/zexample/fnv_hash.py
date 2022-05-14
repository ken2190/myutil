import sys
import struct


def get_byte(b): # python 2
    if sys.version_info[0] == 3:
        return b
    else:
        return ord(b)


def cast_uint64_to_int64(val):
    b = struct.pack("Q", val)
    return struct.unpack("q", b)[0]


def fnv_hash(data, str_encoding="utf-8"):
    if data is None:
        return None
    if isinstance(data, str):
        data = data.encode(str_encoding)
    if not isinstance(data, bytes) and not isinstance(data, unicode):
        raise ValueError("data must be string, bytes or unicode, got " + str(type(data)))

    FNV_SEED = 0x811C9DC5
    FNV64_PRIME = 1099511628211
    FNV64_SIZE = 2**64
    hash = FNV_SEED
    for byte in data:
        hash = ((get_byte(byte) ^ hash) * FNV64_PRIME) % FNV64_SIZE
    hash = cast_uint64_to_int64(hash)
    return hash


TEST_DATA = {
    None                   : None,
    ""                     : 2166136261,
    "hello"                : 6414202926103426347,
    "Hello"                : -7786736384750799925,
    "hello, world!"        : 7205964431561839156,
    "Hello, world!"        : 7098905452484719316,
    "20208810400050001067" : -1473933473301232629,
    "006-ATM-00041067"     : 7223762066447677253,
}


def python_test():
    for k, v in TEST_DATA.items():
        result = fnv_hash(k)
        if result != v:
            print(k, v, result)
            return False
    return True


def pyspark_test():        
    import pyspark
    from pyspark.sql.types import StructType, StructField, LongType, StringType

    def are_dfs_equal(df1, df2):
        if df1.schema != df2.schema:
            return False
        if df1.collect() != df2.collect():
            return False
        return True

    spark = (
        pyspark.sql.SparkSession.builder
        .master('local')
        .appName('fnv_hash_test')
        .config('spark.executor.memory', '1gb')
        .config("spark.cores.max", "1")
        .getOrCreate()
    )

    spark.udf.register("fnv_hash", fnv_hash, LongType())

    def sql_str(val):
        return "'{}'".format(val) if val is not None else "null"

    sql = " union all ".join("select {} as key, fnv_hash({}) as val".format(sql_str(k), sql_str(k)) for k, v in TEST_DATA.items())
    df = spark.sql(sql)

    df.show(len(TEST_DATA), False)

    expected_data = [(k, v) for k, v in TEST_DATA.items()]
    expected_schema = StructType([
        StructField("key", StringType(), True),
        StructField("val", LongType(), True)
    ])
    expected_df = spark.createDataFrame(data=expected_data, schema=expected_schema)

    return are_dfs_equal(df, expected_df)


if __name__ == "__main__":
    python_test_result  = python_test()
    pyspark_test_result = pyspark_test()
    print("python_test",  "OK" if python_test_result  else "FAIL")
    print("pyspark_test", "OK" if pyspark_test_result else "FAIL")
