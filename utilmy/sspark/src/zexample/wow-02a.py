from pyspark.sql.functions import udf, col

# creation of dataframe in previous gist
distinctAuctionItems = df.select("item").distinct().count()
print("Number of distinct items on auction: ", distinctAuctionItems)

getEffectiveBid = udf(lambda col1, col2 : max(col1, col2), LongType())
df = df.withColumn('effectiveBid', getEffectiveBid(col('bid'), col('buyout')))
df = df.select("item", "quantity", "effectiveBid")
df = df.groupBy("item").sum("quantity", "effectiveBid")
df = df.select("item", col("sum(quantity)").alias("cum_quantities"), col("sum(effectiveBid)").alias("cum_bids"))
dfq = df.sort(col("cum_quantities").desc())
dfq.show(3)
dfb = df.sort(col("cum_bids").desc())
dfb.show(3)