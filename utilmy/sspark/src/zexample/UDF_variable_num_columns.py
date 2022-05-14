import pyspark.sql.functions as f
import pyspark.sql.types as t

# ... 

data_frame = data_frame.withColumn('columnB', data_frame['columnA'])
data_frame = data_frame.withColumn('columnC', data_frame['columnA'])
attrs = ['columnA', 'columnB', 'columnC']


# Concatenate the given columns. Each column is of type SparseVector in this case.
def udf_concat_vec(*a):
    result = []
    # a is a tuple of size 1
    var1 = a[0]
    # var1 is a list of size 3
    for var2 in var1:
        result = np.concatenate((result, var2.toArray()))
    return result.tolist()


my_udf_concat_vec = f.UserDefinedFunction(udf_concat_vec, t.ArrayType(t.FloatType()))

data_frame = data_frame.withColumn("together", my_udf_concat_vec(f.array(attrs)))
