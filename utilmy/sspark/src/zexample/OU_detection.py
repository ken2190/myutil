import argparse
import os

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, ArrayType

from build_graph import *
from avg_response_time import *


def datetime_generate(args):
    period = pd.date_range(start=args.start_date, end=args.end_date)
    global DATE_PERIOD
    DATE_PERIOD = period.strftime("date_partition=%Y-%m-%d").tolist()


def load(spark, basePath='/home/kestin/d/soc/data/enron.parquet/', paths=None):
    paths = [paths+dt for dt in DATE_PERIOD]
    paths = [dir_path for dir_path in paths if os.path.isdir(dir_path)]
    df = spark.read.option("basePath",basePath)\
            .parquet(*paths)
    df = df.withColumn('to_account',F.explode(F.col('to_account')))
    print("Get {} rows of data successfully.".format(df.count()))
    return df

def get_employee_count(df):
    member1 = df.select(F.col("from_account").alias("employee")).distinct()
    member2 = df.select(F.col("to_account").alias("employee")).distinct()
    member = member1.union(member2)
    return member


def get_email_count(name):
    return F.udf(lambda x: udf_get_email_count(name, x), IntegerType())
    #df_temp.count()


def udf_get_email_count(name, df):
    df_temp = df.filter((F.col('from_account') == name) or (F.col('to_account') == name))
    return df_temp.count()


import sys
def main(spark, args):
    datetime_generate(args)
    df = load(spark, paths='/home/kestin/d/soc/data/enron.parquet/')
    df.cache()
    # Get all employess in company
    employees = get_employee_count(df)
    employees.cache()

    employees = employees.withColumn(
        'email_cnt', get_email_count(F.col('employee'))(df)
    )
    employees.show()
    sys.exit(0)

    edge_list = gen_edge_list(df)
    edge_list.show(10)
    # edge_list cache suggested
    G = build_graph(edge_list)
    print("Total nodes: {}".format(G.number_of_nodes()))
    print("Total edges: {}".format(G.number_of_edges()))

#    print(G.get_edge_data('chris.germany','tricia.spence'))

#    start = timeit.default_timer()
#    print(nx.shortest_path_length(G,source='chris.germany',target='jeff.dasovich'))
#    end = timeit.default_timer()
#    print(end-start)

    #avg_res_t = get_avg_res_time('kevin.ruscitti', df)
    #avg_res_t = get_avg_res_time('ann.duncan', df)
    #print(avg_res_t)
    from graph_extraction import get_graph_features, get_cliques
    # ans = get_graph_features('kevin.ruscitti', G)
    ans = get_cliques('kevin.ruscitti', G)

    print(*ans)

if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", help="Read email from datetime")
    parser.add_argument("--end_date", help="Read email until datetime")
    args = parser.parse_args()

    spark = SparkSession \
        .builder \
        .appName("feature extraction from parquet") \
        .config("spark.executor.cores", "3") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()


    main(spark, args)
