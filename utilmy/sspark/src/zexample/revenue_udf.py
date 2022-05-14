# for a primer on understanding pandas_udfs
#   https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html


# for understanding decorators see https://realpython.com/primer-on-python-decorators/
# tldr:
#   this:
#     @decorator
#     def func():
#        ...
#   is the same as:
#      def func():
#        ...
#     func = decorator(func)
@pandas_udf('float', PandasUDFType.SCALAR)
def revenue_wo_ship(price: float, quantity: float) -> float:
    return price * quantity

# add a new column using the UDF
sdf.withColumn(
    'revenue_wo_ship',
    revenue_wo_ship(sdf.price, sdf.quantity)
).toPandas()