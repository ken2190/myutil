from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from urllib.parse import urlparse
from pyspark.sql.functions import udf
import os
import fire

def main(input_folder, output_folder):

    spark = SparkSession.builder.config("spark.driver.memory", "16G") .master("local[16]").appName('spark-stats').getOrCreate()
    df = spark.read.parquet(input_folder)

    domain_udf = udf(lambda a:urlparse(a).netloc)

    df_domains = df.select("URL").withColumn("domain", domain_udf(F.col("URL"))).select("domain").groupBy("domain").count().sort(-F.col("count"))
    parquet_folder = output_folder + "/parquet"
    df_domains.repartition(1).write.parquet(parquet_folder)
    # find the parquet file in the folder then move it to domain.parquet
    parquet_folder = output_folder + "/parquet"
    for file in os.listdir(parquet_folder):
        if file.endswith(".parquet"):
            os.rename(parquet_folder + "/" + file, output_folder + "/domains_with_counts.parquet")
    # delete parquet folder with content
    os.system("rm -rf " + parquet_folder)

    # read parquet file, get top 100 and write as csv
    df_domains = spark.read.parquet(output_folder + "/domains_with_counts.parquet")
    csv_folder = output_folder + "/csv"
    df_domains.select("domain", "count").limit(100).repartition(1).write.csv(csv_folder, header=True)

    # find the csv file in the folder then move it to domain.csv
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            os.rename(csv_folder + "/" + file, output_folder + "/top_100_domains.csv")
    # delete csv folder with content
    os.system("rm -rf " + csv_folder)


if __name__ == '__main__':
  fire.Fire(main)


# pip install pyspark fire
# run with python3 domain_stats.py --input_folder "/media/hd/metadata/laion2B-en" --output_folder "/media/hd/domain_stats/laion2B-en"