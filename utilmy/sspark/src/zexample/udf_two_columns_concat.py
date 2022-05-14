import pyspark.sql.functions as f
import pyspark.sql.types as t

# ...
def udf_concat_vec(a, b):
    # a and b of type SparseVector
    return np.concatenate((a.toArray(), b.toArray())).tolist()


my_udf_concat_vec = f.UserDefinedFunction(udf_concat_vec, t.ArrayType(t.FloatType()))

df2 = df.withColumn("togetherAB", my_udf_concat_vec('columnA', 'columnB'))
