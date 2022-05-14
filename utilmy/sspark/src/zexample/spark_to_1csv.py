# Write a spark DataFrame into a single CSV files (to open with Excel/other tools easily)
# Save the file to S3

import s3fs
import pyspark.sql.functions as F  # noqa: N812

def spark_to_csv(spark_df, out_path):
    """
    Save the file in part files with spark and then append them together

    :param spark_df: The spark dataframe to write
    :param out_path: The S3 location where the CSV should be kept
    """
    s3 = s3fs.S3FileSystem(anon=False)
    
    out_path = out_path.rstrip('/')

    drop_names = []
    for name, dtype in spark_df.dtypes:
        if dtype.startswith('map'):
            print('Skipping column:', name, 'as it has dtype:', dtype)
            drop_names.append(name)
        if dtype.startswith('array'):
            spark_df = spark_df.withColumn(name, F.udf(lambda x: str(x))(spark_df[name]))
    spark_df = spark_df.drop(*drop_names)

    print('Writing to {} CSV part files ...'.format(spark_df.rdd.getNumPartitions()))
    # Spark needs s3a:// to write S3 files (in my clusters atleast)
    spark_df.write.format('csv') \
        .option('header', 'false') \
        .option('escape', '"') \
        .save(out_path.replace('s3://', 's3a://') + '_parts', mode='overwrite')

    # Merge the part files into the output
    print('Writing part-files to merged CSV file ...')
    partfiles = sorted(s3.walk(out_path + '_parts' + '/'))
    with s3.open(out_path, 'wb') as fhandler:
        # Add the header as the first row
        fhandler.write((','.join(list(spark_df.columns)) + '\n').encode('utf-8'))
        # Add the rest of the data from the part files
        for ipfile, pfile in partfiles:
            print('Merging part {} of {}'.format(ipfile, len(partfiles)))
            with s3.open(pfile, 'rb') as pfilehandler:
                fhandler.write(pfilehandler.read())

    # Cleanup the temp part file location that was created
    s3.rm(out_path + '_parts', recursive=True)
    print('Final file saved at: {}'.format(out_path))


spark_to_csv(df, 's3://datapath/out.csv')