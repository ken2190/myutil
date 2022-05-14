from os.path import expanduser, join
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark import SparkFiles, SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf

'''On_time UDF'''
def on_time(ts, appid):
    import sys
    import os
    sys.path.append('/usr/lib/python2.7/dist-packages')
    os.system("sudo apt-get install python-numpy -y")
    #os.system("sudo apt-get install google.cloud -y")
    import numpy as np
    import datetime
    import time
    from datetime import timedelta
    from bigtable import insertCell
 
    ts = np.array(ts)
    if ts.size == 0:
        count = 0
        duration = 0
        st = time.mktime(datetime.now())
        ymd = str(datetime.fromtimestamp(st).date())
    else:
    # Genrate one_tag [[start, end], [start, end], ...]
        ts.sort()
        print(ts)
        one_tag = []
        start = float(ts[0])  # start time for this day , that is 1st timestamp to last timestamp array
        for i in range(len(ts)):
            if i == (len(ts)) - 1:           
                end = float(ts[i])
                print("end:-" + str(float(ts[i])))
                a_round = [start, end]  # start and end of array
                one_tag.append(a_round)
            else:
                if float(ts[i+1]) - float(ts[i]) > 3600:                
                    end = float(ts[i])
                    a_round = [start, end]
                    one_tag.append(a_round)
                    start = float(ts[i+1])
                    
                    # Ignore too-short usage
        one_tag = [u for u in one_tag if u[1] - u[0] > 1800]     
    
# Note: Put turn on and turn off time in OneTag Table
        count = int(len(one_tag))
        duration = int(np.diff(one_tag).sum())                  
        ymd = str(datetime.datetime.fromtimestamp(time.time()).date())   
# Note: replace start time of data readed from ORC table and end time which should be included in set.py
        
        insertCell('0003_90T0764589', 5, ymd, 'daily', 'cnt', count)
        insertCell('0003_90T0764590', 5, ymd, 'daily', 'dur', duration)
        insertCell('0003_90T0764591', 5, ymd, 'daily', 'oneTag', one_tag)
    return {'count':count,'duration':duration, 'ymd':ymd}
   
def split_byApTypes(df, appid):
    df_app = df.filter(df.apptypeid == appid)
    return df_app

def apply_ThresholdPower(df, threshold_power):
    df_pow = df.filter(df.powers > threshold_power)
    return df_pow
    
    
def main(sc,spark):
    #dataframe which contains specific duration dtaa: ex. last 24 hours power usage
    df_src = spark.sql("select userid, apptypeid, ts, powers, fixed_at from umbrella.hourly_users_orc where ts <= fixed_at")
    df_dist_drop_fixed = df_src.drop(df_src.fixed_at).distinct()
    df_dist_drop_fixed.persist()

    udf_on_time=udf(on_time, MapType(StringType(),StringType(), valueContainsNull=True))
    udf_on_time=udf(on_time,StringType())
    import set
    reload(set)
    
    for appid in set.app_type_list:
        df_appid = split_byApTypes(df_dist_drop_fixed, appid)
        df_passed_app_powers = apply_ThresholdPower(df_appid, set.thresholds[str(appid)][0])
        df_dist = df_passed_app_powers.drop(df_passed_app_powers.powers).distinct()
        df_g = df_dist.groupBy('userid','apptypeid')
        df_ts_list = df_g.agg(collect_list('ts').alias('ts_list'))
        df_ts_list.persist()
        df_ts_list.withColumn("onew_tag", udf_on_time("ts_list",df_ts_list.apptypeid)).show(10)
    print(type(df_ts_list))

    
if __name__ == "__main__":
    warehouse_location = 'file:${system:user.dir}/spark-warehouse'
    path = "gs:///set.py"
    print(path)
    sc = SparkContext()
    sc.addFile(path)
    sc.addFile("gs:///bigtable.py")
    sc.addPyFile(path)
    spark = SparkSession \
    .builder.master("yarn") \
    .appName("Python Spark SQL Hive integration") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()
    main(sc, spark)
    sc.stop()