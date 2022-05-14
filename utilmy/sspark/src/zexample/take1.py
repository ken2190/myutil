from pyspark.sql.functions import udf, col

search = SearchEngine()

@udf('string')
def get_zip_udf1(latitude, longitude):
    try:
        zip = search.by_coordinates(latitude, longitude, returns=1)[0].to_dict()["zipcode"]
    except:
        zip = 'bad'
    return zip


df.withColumn('zip', get_zip_udf1(col("latitude"),col("longitude"))).show()