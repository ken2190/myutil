from pyspark.sql.functions import col

data = data.select(col("Name").alias("name"), col("askdaosdka").alias("age"))
data.show()

# Output
#+-------+---+
#|   name|age|
#+-------+---+
#|Alberto|  2|
#| Dakota|  2|
#+-------+---+