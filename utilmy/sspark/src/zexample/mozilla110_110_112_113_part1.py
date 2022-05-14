import sys
import datetime
import random
import subprocess
import mozillametricstools.common.functions as mozfun

# "active_addons"
mozfun.register_udf(sqlContext
                    , lambda arr:  sum(arr) if arr else 0, "array_sum"
                    , pyspark.sql.types.IntegerType())
ms = sqlContext.read.load("s3://telemetry-parquet/main_summary/v4", "parquet"
                          , mergeSchema=True)
sqlContext.registerDataFrameAsTable(ms,"ms")


#



ms3 = sqlContext.sql("""
   select
   sample_id,client_id, distribution_id,subsession_start_date,
   profile_creation_date,default_search_engine,
   subsession_length, scalar_parent_browser_engagement_total_uri_count , is_default_browser,
   scalar_parent_browser_engagement_tab_open_event_count, search_counts
   from ms
   where
   distribution_id in ( 'mozilla111','mozilla112','mozilla113','mozilla110')
   and profile_creation_date >= 17292 and profile_creation_date <= 17306
   and substring(submission_date, 1,10)>='2017-05-07'
   and channel = 'release'
   and app_name = 'Firefox'
""")
sqlContext.registerDataFrameAsTable(ms3, "ms3")
# sqlContext.sql("""
# select distribution_id, count(distinct(client_id))*4
# from ms3 where sample_id <='25'  group by distribution_id 
# """).toPandas()





ms4 = sqlContext.sql("""
   select 
   client_id as cid,
   case 
       when distribution_id is null then 'miss' 
       else distribution_id
   end as did,
   substring(subsession_start_date,1,10) as date,
   from_unixtime(profile_creation_date*86400,'yyyy-MM-dd') as pcd,
   count(*) as ns,
   last(case when default_search_engine is not null 
            then default_search_engine else 'miss' end) as seng,
   sum(case when subsession_length is null 
            then 0 else subsession_length end) as ssl,
   sum(case when  scalar_parent_browser_engagement_total_uri_count is null 
            then  0 else  scalar_parent_browser_engagement_total_uri_count end ) as turi,
   max(case when is_default_browser is null 
            then False else is_default_browser end) as isdef,
   sum(case when  scalar_parent_browser_engagement_tab_open_event_count is null 
            then 0 else  scalar_parent_browser_engagement_tab_open_event_count end) as ttabs,
   sum(case 
     when search_counts is not null then array_sum(search_counts.count) else 0 
   end)  as tsearch
   from ms3
   group by 1,2,3,4
""")
