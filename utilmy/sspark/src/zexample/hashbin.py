class HashBin(object):
    def __init__(self, exp_for_2):
        assert exp_for_2 > -1
        assert isinstance(exp_for_2, int)
        
        self.exp_for_2 = exp_for_2
    
    @property
    def nbins(self):
        return 2**self.exp_for_2
    
    def bin_num_iter(self):
        return range(self.nbins)
    
    def get_bin_num(self, obj):
        return hash(obj) & (self.nbins - 1)

def get_bin_num(obj, nbins=32):
    import math
    
    # nbins must be a power of 2
    assert (math.log(nbins, 2) % 1) == 0
    
    return hash(obj) & (nbins - 1)


def spark_df_to_pandas_chunked(spark_df, hash_col, nbins=32):
    import pandas
    import math
    from pyspark.sql.functions import udf, lit, col
    from pyspark.sql.types import IntegerType
    
    # nbins must be a power of 2
    assert (math.log(nbins, 2) % 1) == 0
    get_bin_num_udf = udf(get_bin_num, IntegerType())
    
    pd_dfs = (
        spark_df.filter(get_bin_num_udf(col(hash_col), lit(nbins)) == bin_).toPandas()
        for bin_ in range(nbins)
    )
    return pandas.concat(pd_dfs, axis=0)
