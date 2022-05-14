import argparse
import numpy as np

from pyspark.sql.types import ArrayType, StringType, IntegerType, StructType, StructField
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

# establish spark connection
spark = (
    SparkSession.builder
    .master('local[*]')
    .getOrCreate()
)


@f.udf(ArrayType(IntegerType()))
def generate_numbers(start: int, end: int) -> list:
    """Generating numbers between a lower and upper boundary"""
    return np.arange(start, end + 1).tolist()


@f.udf(ArrayType(ArrayType(IntegerType())))
def reshape_list(a: list) -> list:
    """Reshaping list into a n by 3 array of array"""
    a = a[: - (len(a) % 3)] if len(a) % 3 else a
    return np.reshape(a, (int(len(a) / 3), 3)).tolist()


@f.udf(StructType([
    StructField('gene_id', StringType(), False),
    StructField('transcript_id', StringType(), False),
    StructField('gene_name', StringType(), False),
    StructField('protein_id', StringType(), False)
]))
def parse_annotation(annotation: str) -> StructType:
    """Parsing gff3 sequence annotation to get a dictionary of values"""

    columns = ['transcript_id', 'gene_id', 'gene_name', 'protein_id']
    return {
        feature.split('=')[0]: feature.split('=')[1] for feature in annotation.split(';') if feature.split('=')[0] in columns
    }


def main(gencode_file: str, output_file: str) -> None:

    # Helper function to remove version number from gene_id:
    remove_version = lambda column: f.regexp_replace(column, r'\.\d+', '')

    # Schema for the GENCODE file:
    gff3_schema = StructType([
        StructField("chr", StringType(), False),
        StructField("source", StringType(), False),
        StructField("featureType", StringType(), False),
        StructField("start", IntegerType(), False),
        StructField("end", IntegerType(), False),
        StructField("score", StringType(), False),
        StructField("strand", StringType(), False),
        StructField("phase", StringType(), False),
        StructField("annotation", StringType(), False),
    ])

    # Open and parse the GENCODE file:
    protein_coding_segments = (
        spark.read.option("comment", "#").csv(gencode_file, sep='\t', schema=gff3_schema)

        # Filter for coding sequences only:
        .filter(f.col('featureType') == 'CDS')

        # Parsing GFF3 annotation:
        .withColumn('parsed_annotation', parse_annotation(f.col('annotation')))

        # Selecting columns + extract fields:
        .select('chr', 'start', 'end', 'strand', 'parsed_annotation.*')
    )

    # Appling further processing steps on the dataframe:
    processed_df = (
        protein_coding_segments

        # Cleaning identifiers from version information:
        .withColumn('gene_id', remove_version(f.col('gene_id')))
        .withColumn('transcript_id', remove_version(f.col('transcript_id')))
        .withColumn('protein_id', remove_version(f.col('protein_id')))

        # Order dataframe:
        .orderBy(['chr', 'start'])
    )

    # Generating positions and saving to file:
    protein2gene_mapping = (
        processed_df

        # Generating positions for all CDS fragments:
        .withColumn('positions', generate_numbers(f.col('start'), f.col('end')))

        # Grouping by protein_id
        .groupBy('protein_id')
        .agg(
            f.first('chr').alias('chr'),
            f.first('strand').alias('strand'),
            f.first('gene_id').alias('gene_id'),
            f.first('transcript_id').alias('transcript_id'),
            f.first('gene_name').alias('gene_name'),
            f.flatten(f.collect_set('positions')).alias('positions')
        )
        .withColumn(
            'positions',
            f.when(f.col('strand') == '+', f.col('positions'))
            .otherwise(f.reverse(f.col('positions')))
        )
        .withColumn('positions', reshape_list(f.col('positions')))
        .withColumn('bases', f.explode(f.expr("""transform(positions,(x,i)-> struct(x as position,(i+1) as position_number))""")))
        .select('*', 'bases.*')
        .drop('positions', 'bases')
    )

    # Let's do some test. If any of the test fails, the program will stop and the data will not be saved.
    assert protein2gene_mapping


    # If all looks good, save dataset:
    protein2gene_mapping.write.mode('overwrite').parquet(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tool to generate mappings between protein position and genomic location.')
    parser.add_argument('-g', '--gencode_file', type=str, help='Gencode file.', required=True)
    parser.add_argument('-o', '--output_file', type=str, help='Output parquet file.', required=True)

    args = parser.parse_args()

    main(args.gencode_file, args.output_file)
