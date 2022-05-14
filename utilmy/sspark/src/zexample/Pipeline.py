from pyspark.sql import SparkSession
import geopandas as gpd
from pyspark.sql import functions as f
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from shapely.geometry import Point
from decimal import Decimal
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

bucket = "your_bucket"
spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('spatial-data-clustering') \
  .getOrCreate()

# Load data from BigQuery.
df_hires = spark.read.format('bigquery') \
  .option('table', 'bigquery-public-data:london_bicycles.cycle_hire') \
  .load()
df_stations = spark.read.format('bigquery') \
  .option('table', 'bigquery-public-data:london_bicycles.cycle_stations') \
  .load()

# Read LSOA boundaries data from ESRI folder within the upper bucket
lsoa = gpd.read_file('gs://your_bucket/ESRI/LSOA_2011_London_gen_MHW.shp')
lsoa.to_crs({'init': 'epsg:4326'},inplace=True)

# Join and aggregate data from Bigquery
df_hires.createOrReplaceTempView("hires")
df_stations.createOrReplaceTempView("stations")
joined_hires_stations = spark.sql(
    """
WITH joined_hires AS (
      SELECT
         h.start_station_name as station_name,
         IF(EXTRACT(DAYOFWEEK FROM h.start_date) = 1 OR
            EXTRACT(DAYOFWEEK FROM h.start_date) = 7, "weekend", "weekday") as start_dayofweek,
         h.duration,
         s.bikes_count,
         s.latitude,
         s.longitude
     FROM hires as h
     JOIN stations as s
     ON h.start_station_id = s.id
                      ),
    aggregated_hires AS (
      SELECT
         station_name,
         start_dayofweek,
         AVG(duration) as average_duration,
         COUNT(*) as number_hires,
         MAX(bikes_count) as bikes_per_station,
         MAX(longitude) as longitude,
         MAX(latitude) as latitude
     FROM joined_hires
     GROUP BY station_name, start_dayofweek
                        )
SELECT *
FROM aggregated_hires
ORDER BY station_name
    """
)
joined_hires_stations = joined_hires_stations.withColumn('duration_weekday', f.when(f.col('start_dayofweek') == "weekday", f.col('average_duration')).otherwise(0))
joined_hires_stations = joined_hires_stations.withColumn('duration_weekend', f.when(f.col('start_dayofweek') == "weekend", f.col('average_duration')).otherwise(0))
df_stations_stats = joined_hires_stations.groupby('station_name').agg(f.sum('duration_weekday').alias('wday_duration_per_station'),
                                                 f.sum('duration_weekend').alias('wend_duration_per_station'),
                                                 f.sum('number_hires').alias('hires_per_station'),
                                                 f.max('bikes_per_station').alias('bikes_per_station'),
                                                 f.max('longitude').alias('longitude'),
                                                 f.max('latitude').alias('latitude')
                                                 )
# Spatial join between point ant polygon datasets
b_lsoa = spark.sparkContext.broadcast(lsoa)
def find_lsoa(longitude, latitude):
    is_lsoa = b_lsoa.value.apply(lambda x: x['LSOA11NM']  if x['geometry'].intersects(Point(Decimal(longitude), Decimal(latitude))) else None, axis=1)
    valid_idx = is_lsoa.first_valid_index()
    lsoa_name = is_lsoa.loc[valid_idx] if valid_idx is not None else None
    return  lsoa_name
find_lsoa_udf = udf(find_lsoa, StringType())
df_with_lsoa = df_stations_stats.withColumn('lsoa', find_lsoa_udf(col('longitude'), col('latitude')))

# Group data by lSOAs
df_lsoa_stats = df_with_lsoa.groupby('lsoa').agg(f.avg('wday_duration_per_station').alias('tot_wday_duration'),
                                                 f.avg('wend_duration_per_station').alias('tot_wend_duration'),
                                                 f.sum('hires_per_station').alias('tot_hires'),
                                                 f.sum('bikes_per_station').alias('tot_bikes')
                                                 )

# MLlib k-means clustering
vecAssembler = VectorAssembler(inputCols=["tot_wday_duration", "tot_wend_duration", "tot_hires", "tot_bikes"], outputCol="features")
training_df = vecAssembler.transform(df_lsoa_stats)
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(training_df.select('features'))
transformed = model.transform(training_df)

# show clusters
transformed.show()
