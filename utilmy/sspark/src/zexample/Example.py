# Does not work because bc is not yet defined (line 19) when the function is defined (line 9)

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, lower, lit, array
from pyspark.sql.types import StringType, ArrayType

def test(cc):
    return bc.value + ' ' + cc + " bar"

test = udf(test, StringType())

if __name__ == "__main__":
    # Setup spark
    spark = SparkSession.builder \
        .appName('Geomap') \
        .getOrCreate()
    bc = spark.sparkContext.broadcast('baz')

    parser = argparse.ArgumentParser(description='Create tables for countries')
    parser.add_argument('file')
    args = parser.parse_args()

    df = spark.read.csv(args.file, header=True)
    df = df.withColumn('foo bar', test(df.foo))
    df.write.csv('test_foo.csv')


# Does not work because the test function is invoked (line 50) before it is defined (line 53)

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, lower, lit, array
from pyspark.sql.types import StringType, ArrayType

if __name__ == "__main__":
    # Setup spark
    spark = SparkSession.builder \
        .appName('Geomap') \
        .getOrCreate()
    bc = spark.sparkContext.broadcast('baz')

    parser = argparse.ArgumentParser(description='Create tables for countries')
    parser.add_argument('file')
    args = parser.parse_args()

    df = spark.read.csv(args.file, header=True)
    df = df.withColumn('foo bar', test(df.foo))
    df.write.csv('test_foo.csv')

def test(cc):
    return bc.value + ' ' + cc + " bar"

test = udf(test, StringType())


# Works because you define the bc variable (line 72) before the function that uses it (line 78), 
# and you define the function before you use it (line 85)

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, lower, lit, array
from pyspark.sql.types import StringType, ArrayType

if __name__ == "__main__":
    # Setup spark
    spark = SparkSession.builder \
        .appName('Geomap') \
        .getOrCreate()
    bc = spark.sparkContext.broadcast('baz')

    parser = argparse.ArgumentParser(description='Create tables for countries')
    parser.add_argument('file')
    args = parser.parse_args()

    def test(cc):
        return bc.value + ' ' + cc + " bar"

    test = udf(test, StringType())

    df = spark.read.csv(args.file, header=True)
    df = df.withColumn('foo bar', test(df.foo))
    df.write.csv('test_foo.csv')
    
    
# Also works because the test function is explicitly provided to the main function as a parameter

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, lower, lit, array
from pyspark.sql.types import StringType, ArrayType

def main(spark, func):
    parser = argparse.ArgumentParser(description='Create tables for countries')
    parser.add_argument('file')
    args = parser.parse_args()

    df = spark.read.csv(args.file, header=True)
    df = df.withColumn('foo bar', func(df.foo))
    df.write.csv('test_foo.csv')


if __name__ == "__main__":
    # Setup spark
    spark = SparkSession.builder \
        .appName('Geomap') \
        .getOrCreate()
    bc = spark.sparkContext.broadcast('baz')

    def test(cc):
        return bc.value + ' ' + cc + " bar"

    test = udf(test, StringType())

    main(spark, test)