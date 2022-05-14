

import click
import tldextract

from pyspark.sql import Row, types as T
from pyspark.sql.functions import udf

from geovec_data.utils import get_spark, update_src
from geovec_data import paths


TLD_SCHEMA = T.StructType([
    T.StructField('subdomain', T.StringType()),
    T.StructField('domain', T.StringType()),
    T.StructField('suffix', T.StringType()),
    T.StructField('registered_domain', T.StringType()),
])


@udf(T.ArrayType(TLD_SCHEMA))
def extract_tlds(urls):
    """Parse TLDs from URLs.
    """
    tlds = []
    for url in urls:

        try:

            tld = tldextract.extract(url.lower())

            tlds.append(Row(
                subdomain=tld.subdomain,
                domain=tld.domain,
                suffix=tld.suffix,
                registered_domain=tld.registered_domain,
            ))

        except Exception as e:
            pass

    return tlds or None


@click.command()
@click.option('--src', type=str, default=paths.TWEETS_DST)
def main(src):
    """Extract link domains.
    """
    sc, spark = get_spark()

    tweets = spark.read.parquet(src)

    tlds = extract_tlds(tweets.twitter_entities.urls.expanded_url)

    tweets = tweets.withColumn('tlds', tlds)

    update_src(tweets, src)


if __name__ == '__main__':
    main()
