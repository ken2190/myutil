# PySpark DataFrame UDFs using a whole row

import pyspark.sql.functions as f

def my_func(row):
    '''
    An example use case - take a whole row, combine lots of column values with some custom function
    In this case, pluck out two values and return some JSON for them
    '''
    my_dict = {
        "col1": row.col1,
        "col2": row.col2
    }
    import json
    return json.dumps(my_dict)

my_func_udf = f.udf(lambda r: my_func(r), StringType())

documents = df.select(
    my_func_udf(
        f.struct([df[x] for x in df.columns])
    ).alias('json')
).show()