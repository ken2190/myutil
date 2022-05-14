from pyspark.sql.functions import udf, col


@udf('string')
def get_zip_udf3(latitude, longitude):
    search =  SearchEngine(db_file_dir="/tmp/db")
    try:
        zip = search.by_coordinates(latitude, longitude, returns=1)[0].to_dict()["zipcode"]
    except:
        zip = 'bad'
    return zip

  
df.withColumn('zip', get_zip_udf3(col("latitude"),col("longitude"))).show()