def main():
  
  # Import data
  df = spark.read.parquet("dbfs:/databricks-datasets/amazon/test4K/part-r-00000-64a9bd4a-25fc-48e6-8a60-2fd057bddd27.gz.parquet")
  # Select relevant columns
  dfSelect = df.select("asin", "review")
  # Using text analytics to start analysing word frequency
  # To standardise responses, convert them all to lowercase
  dfLower = dfSelect.withColumn("review_lower", lower(col("review")))
  # Remove the usual punctuations so we don't include them into our analysis
  dfReplace = dfLower.withColumn("review_replace", regexp_replace(col("review_lower"), r'[.,!]', ' '))
  # Replace "svc" with "service" as part of standardisation
  dfReplace = dfReplace.withColumn("review_replace", regexp_replace(col("review_replace"), 'svc', 'service'))
  # Apply lemmatisation, which also tokenises responses in the process
  dfNounPhrase = dfReplace.withColumn("review_nounphrase", udfNounPhrase("review_replace"))
  # For every noun phrase, put it into another row, instead of sharing a row with multiple other noun phrases in an array
  dfExplode = dfNounPhrase.withColumn("review_split", explode(col("review_nounphrase")))
  # Remove rows with empty tokens or with a single character token
  dfFilter = dfExplode.withColumn("review_length", length(col("review_split")))
  dfFilter = dfFilter.filter(col("review_length") > 1)
  # Let's count the number of unique users who mentioned each token (this is the word frequency table)
  dfFrequency = dfFilter.groupBy(col("review_split")).agg(count(col("review")).alias("reviews"))
  # Show popular noun phrases
  dfFrequency.createOrReplaceTempView("dffrequency")
  print("Retrieving popular noun phrases...")
  spark.sql("SELECT review_split, SUM(reviews) AS sum_reviews FROM dffrequency GROUP BY review_split ORDER BY sum_reviews DESC").show(truncate = False)
  
  return dfFrequency