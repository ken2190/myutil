#!/usr/bin/env python3

import argparse
import os
import sys

import oci
import oci_dataflow
from pyspark import SparkConf
from pyspark.sql.functions import udf
from pyspark.sql.types import SparkSession, StringType


def am_in_dataflow():
    if os.environ.get("HOME") == "/home/dataflow":
        return True
    return False


def get_dataflow_spark_session(file_location=None, profile_name=None):
    if am_in_dataflow():
        spark = SparkSession.builder.appName("adw_example").getOrCreate()
    else:
        # Import OCI.
        try:
            import oci
        except:
            raise Exception(
                "You need to install the OCI python library to use oci_dataflow locally"
            )

        # Use defaults for anything unset.
        if file_location is None:
            file_location = oci.config.DEFAULT_LOCATION
        if profile_name is None:
            profile_name = oci.config.DEFAULT_PROFILE

        # Load the config file.
        try:
            oci_config = oci.config.from_file(
                file_location=file_location, profile_name=profile_name
            )
        except Exception as e:
            print(
                "You need to set up your OCI config properly to use oci_dataflow locally"
            )
            raise e
        conf = SparkConf()
        conf.set("fs.oci.client.auth.tenantId", oci_config["tenancy"])
        conf.set("fs.oci.client.auth.userId", oci_config["user"])
        conf.set("fs.oci.client.auth.fingerprint", oci_config["fingerprint"])
        conf.set("fs.oci.client.auth.pemfilepath", oci_config["key_file"])
        conf.set(
            "fs.oci.client.hostname",
            "https://objectstorage.{0}.oraclecloud.com".format(oci_config["region"]),
        )
        spark = (
            SparkSession.builder.appName("adw_example").config(conf=conf).getOrCreate()
        )
    return spark


def get_authenticated_client(spark, client):
    import oci

    if os.environ.get("HOME") != "/home/dataflow":
        # We are running locally, use our API Key.
        config = oci.config.from_file()
        authenticated_client = client(config)
    else:
        # We are running in Data Flow, use our Delegation Token.
        conf = spark.sparkContext.getConf()
        token_path = conf.get("spark.hadoop.fs.oci.client.auth.delegationTokenPath")
        with open(token_path) as fd:
            delegation_token = fd.read()
        signer = oci.auth.signers.InstancePrincipalsDelegationTokenSigner(
            delegation_token=delegation_token
        )
        authenticated_client = client(config={}, signer=signer)
    return authenticated_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bucket", default="input_sample_data")
    parser.add_argument("--output-bucket", default="output_sample_data")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    input_bucket = args.input_bucket
    output_bucket = args.output_bucket
    target_file = "joined.csv"
    if args.reset:
        print("Resetting output data.")
        command = f"oci os object bulk-delete --bucket-name {output_bucket} --prefix {target_file} --force"
        retval = os.system(command)
        sys.exit(retval)

    # Get our Spark session.
    spark = oci_dataflow.get_dataflow_spark_session()

    # Get our OCI Object Storage Namespace.
    client = oci_dataflow.get_authenticated_client(
        spark, oci.object_storage.ObjectStorageClient
    )
    namespace = client.get_namespace().data

    # Generate URIs of our CSV files.
    files = ["auto-mpg.csv", "manufacturers.csv"]
    uris = {file: f"oci://{input_bucket}@{namespace}/{file}" for file in files}

    # Load our DataFrames.
    print("Loading MPG information from " + uris["auto-mpg.csv"])
    auto_mpg_df = (
        spark.read.format("csv").option("header", True).load(uris["auto-mpg.csv"])
    )
    print("Loading manufacturer information from " + uris["manufacturers.csv"])
    manufacturers_df = (
        spark.read.format("csv").option("header", True).load(uris["manufacturers.csv"])
    )

    # Add a manufacturers column, to join with the manufacturers list.
    first_word_udf = udf(lambda x: x.split()[0], StringType())
    auto_mpg_df = auto_mpg_df.withColumn(
        "manufacturer", first_word_udf(auto_mpg_df.carname)
    )

    # Join the DataFrames.
    joined = auto_mpg_df.join(manufacturers_df, "manufacturer")

    # Output the results.
    output_uri = f"oci://{output_bucket}@{namespace}/joined.csv"
    print("Writing joined DataFrame to " + output_uri)
    joined.coalesce(1).write.csv(output_uri, header="true")
    print("Wrote {} rows".format(joined.count()))


main()
