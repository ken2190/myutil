import pyspark.sql.functions as F
import pyspark.sql.types as T

# Create data frame
customers = spark.createDataFrame(data=[[["Alice", "Bob"]], [["Alice", "Amanda", "John"]], [["Alice", "Christ", "Bryan", "Adam"]],  [["Alice"]], [["Bob"]], [["Bob", "Cynthia"]], [["Bob", "Bale", "Anita"]], [["Bob"]]], schema=["name"])

# Create UDF
def lower_case(x):
    res = []
    for x_ in x:
        res.append(x_.lower())
    return res

convert_to_lower = F.udf(lower_case, T.ArrayType(T.StringType()))

# Apply UDF
customers = customers.withColumn("new_name", convert_to_lower(F.col("name")))

# Create new data
new_customers = spark.createDataFrame(data=[["Karen"], ["Penny"], ["John"], ["Cosimo"]], schema=["name"])

# Convert string to array of string
new_customers = new_customers.withColumn("new_name", F.array(F.col("name")))

# Apply UDF
new_customers = new_customers.withColumn("new_name", convert_to_lower(F.col("new_name")))

# See 10 rows from results without truncation when printing
new_customers.show(10, truncate=False)
