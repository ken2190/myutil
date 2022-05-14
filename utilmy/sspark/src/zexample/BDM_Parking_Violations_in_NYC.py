from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import sys

# constants
VIOLATIONS_FILEPATH = 'hdfs:///tmp/bdm/nyc_parking_violation/*'
CSCL_FILEPATH = 'hdfs:///tmp/bdm/nyc_cscl.csv'
YEARS = list(range(2015, 2020))
COUNTY_MAP = {'MAN': '1','MH': '1','MN': '1','NEWY': '1','NEW Y': '1','NY': '1',
              'BRONX': '2', 'BX': '2',
              'BK': '3', 'K': '3', 'KING': '3', 'KINGS': '3',
              'Q': '4', 'QN': '4', 'QNS': '4', 'QU': '4', 'QUEEN': '4',
              'R': '5', 'RICHMOND': '5'}
HSE_COLS = ['L_LOW_HN', 'L_HIGH_HN', 'R_LOW_HN', 'R_HIGH_HN']
CSCL_COLS = ['PHYSICALID', 'BOROCODE', 'FULL_STREE', 'ST_LABEL'] + HSE_COLS


# get lin reg coefficients using statsmodels
def get_coef(y1, y2, y3, y4, y5):
    import numpy as np
    import statsmodels.api as sm

    x = np.array(YEARS)
    X = sm.add_constant(x)
    y = np.array([y1,y2,y3,y4,y5])

    return float(sm.OLS(y,X).fit().params[1])


if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)

    ############### 1. Preparing violations data
    # reading violations data
    vio = spark.read.csv(VIOLATIONS_FILEPATH, header=True)\
            .select('Issue date', 'Street Name', 'House Number', 'Violation County')\
            .dropna(how='any').cache()

    # condition for year 2015 - 2019
    vio = vio.withColumn('year', F.to_date(vio['Issue date'], 'mm/dd/yyyy'))
    vio = vio.withColumn('year', F.year(vio.year))
    cond_year = vio.year.isin(YEARS)

    # condition for hse numbers: (#), (# #), (#-#)
    cond_hn = vio['House Number'].rlike('^[0-9]+([ -][0-9]+)?$')

    # remove those w invalid years or hse numbers
    vio = vio.filter(cond_year & cond_hn)

    # rename columns
    vio = vio.select('year', 
                   F.col('Violation County').alias('county'),
                   F.col('Street Name').alias('st_name'),
                   F.col('House Number').alias('hse_num'))

    # uppercase street names
    vio = vio.withColumn('st_name', F.upper(vio.st_name))

    # replace county with boros
    vio = vio.replace(COUNTY_MAP, subset='county')

    # split house number by space or hyphen
    vio = vio.withColumn('hse_num', F.split(vio.hse_num, ' |-'))

    # get total counts by location and year
    vio = vio.groupby(vio.columns).count()

    # pivot by year, to reduce number of rows
    vio = vio.groupby('county', 'st_name', 'hse_num').pivot('year', YEARS).agg(F.max('count'))

    # type cast arraytype to int
    vio = vio.withColumn('hse_num', F.expr('transform(hse_num, x-> int(x))'))


    ############### 2. Preparing centerline data
    # read file
    cscl = spark.read.csv(CSCL_FILEPATH, header=True).select(CSCL_COLS).cache()
    cscl = cscl.withColumn('PHYSICALID', cscl.PHYSICALID.astype('int'))
    for hse_col in HSE_COLS:
        # split by delimiter
        cscl = cscl.withColumn(hse_col, F.split(hse_col, ' |-'))
        # type cast HN to int
        cscl = cscl.withColumn(hse_col, F.expr('transform(%s, x-> int(x))'%hse_col))

    # consolidate FULL_STREE and ST_LABEL into single column
    cscl = cscl.groupby('PHYSICALID')\
        .agg(F.array_join(F.collect_set('FULL_STREE'), '_').alias('full_st'),
             F.array_join(F.collect_set('ST_LABEL'), '_').alias('st_label'),
             F.first('L_LOW_HN').alias('L_LOW_HN'),
             F.first('L_HIGH_HN').alias('L_HIGH_HN'),
             F.first('R_LOW_HN').alias('R_LOW_HN'),
             F.first('R_HIGH_HN').alias('R_HIGH_HN'),
             F.first('BOROCODE').alias('BOROCODE'),
            )
    cscl = cscl.withColumn('st', F.array_distinct(F.split(F.concat_ws('_', 'full_st', 'st_label'), '_'))) 


    ############### 3. Join violations to centerline
    # condition for boro 
    cond_boro = (vio.county == cscl.BOROCODE)

    # condition for street name
    cond_st = F.expr('array_contains(st, st_name)')

    # condition for house number
    subcond_even = ((F.element_at(vio.hse_num, -1)%2==0)
                    & (F.size(vio.hse_num) == F.size(cscl.R_LOW_HN))
                    & (F.element_at(vio.hse_num, -1) >= F.element_at(cscl.R_LOW_HN, -1))
                    & (F.element_at(vio.hse_num, -1) <= F.element_at(cscl.R_HIGH_HN, -1))
                    & (((F.size(vio.hse_num) == 2) & (vio.hse_num[0] == cscl.R_LOW_HN[0]))
                       | (F.size(vio.hse_num) == 1))
                    )
    subcond_odd = ((F.element_at(vio.hse_num, -1)%2!=0)
                    & (F.size(vio.hse_num) == F.size(cscl.L_LOW_HN))
                    & (F.element_at(vio.hse_num, -1) >= F.element_at(cscl.L_LOW_HN, -1))
                    & (F.element_at(vio.hse_num, -1) <= F.element_at(cscl.L_HIGH_HN, -1))
                    & (((F.size(vio.hse_num) == 2) & (vio.hse_num[0] == cscl.L_LOW_HN[0]))
                       | (F.size(vio.hse_num) == 1))
                    )
    cond_hn = (subcond_even | subcond_odd)

    # actual join
    joined = vio.join(F.broadcast(cscl), [cond_boro, cond_hn, cond_st])\
                 .select('PHYSICALID', '2015', '2016', '2017', '2018', '2019')\
                 .fillna(0).cache()

    # aggregate counts by phy id
    joined = joined.groupby('PHYSICALID')\
        .agg(F.sum('2015').alias('2015'),
             F.sum('2016').alias('2016'),
             F.sum('2017').alias('2017'),
             F.sum('2018').alias('2018'),
             F.sum('2019').alias('2019'))

    # union with distinct phy ids to recover phyids with no violations
    distinct_cscl = cscl.select('PHYSICALID')\
                        .distinct().alias('distinct_ids')\
                        .cache()
    distinct_cscl = distinct_cscl.withColumn('2015', F.lit(0))\
                                .withColumn('2016', F.lit(0))\
                                .withColumn('2017', F.lit(0))\
                                .withColumn('2018', F.lit(0))\
                                .withColumn('2019', F.lit(0))
    joined = joined.union(distinct_cscl)
    joined = joined.groupby('PHYSICALID')\
          .agg(F.max('2015').alias('2015'),
               F.max('2016').alias('2016'),
               F.max('2017').alias('2017'),
               F.max('2018').alias('2018'),
               F.max('2019').alias('2019'))


    ############### 4. Linear regression
    get_coef_udf = F.udf(get_coef, FloatType())
    joined = joined.withColumn('coef', get_coef_udf(joined['2015'],
                                                    joined['2016'],
                                                    joined['2017'],
                                                    joined['2018'], 
                                                    joined['2019']))

    out_folder = sys.argv[1]
    joined.orderBy('PHYSICALID').write.csv(out_folder, mode='overwrite')
