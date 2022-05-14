from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F
from pyspark.sql import Window as W


def add_season_of_year(features, date_col: str = 'date'):
    month2season = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
        12: 0,
    }
    add_season_udf = F.udf(lambda x: month2season[x], IntegerType())
    features = features.withColumn('season', add_season_udf(F.month(date_col)))
    return features

def complite_features(features):
    w = W.partitionBy('store', 'plu').orderBy('ts')
    w_desc = W.partitionBy('store', 'plu').orderBy(F.col('ts').desc())

    features = (
        features
        .withColumn('promo_ind', ((~F.isnull('promotionid')) & (F.col('promotionid') != '')).cast('byte'))
        .withColumn('ts', F.to_timestamp('enddatetimestamp', 'yyyyMMddHHmmss').cast("long"))
        .withColumn('dt', F.to_date('enddatetimestamp', 'yyyyMMddHHmmss'))
        .where(F.col('ts').isNotNull())
        .select('plu', 'store',  'promo_ind', 'base_qty', 'price', 'ts', 'dt')
        .withColumn('prev_ts', F.lead('ts', 1).over(w_desc))
        .withColumn('next_ts', F.lead('ts', 1).over(w))
        .withColumn('seconds_to_prev', F.col('ts') - F.col('prev_ts'))
        .withColumn('seconds_to_next', F.col('next_ts')- F.col('ts'))
        .filter(F.col('seconds_to_next') > 0)
        .withColumn('weekday', F.dayofweek('dt'))
        .withColumn('is_weekend', F.col('weekday').isin([1, 7]).astype('int'))
        .orderBy('store', 'plu', 'ts')
    )

    features = add_season_of_year(features, date_col='dt')
    return features
