from urllib.parse import urlparse
from pyspark.sql.functions import udf

domain_udf = udf(lambda a:urlparse(a).netloc)

df_domains = df.select("URL").withColumn("domain", domain_udf(F.col("URL"))).select("domain").groupBy("domain").count().sort(-F.col("count"))
df_domains.repartition(1).write.parquet("domains")