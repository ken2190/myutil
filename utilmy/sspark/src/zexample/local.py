from pyspark.sql import Row, SQLContext, HiveContext
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import udf, col
import numpy as np

hiveContext = HiveContext(sc)

# load a local copy of the S3 output data
df = hiveContext.read.json("./from_qubole_2") 

# create a pandas dataframe and clean up the column names
dfp = df.toPandas() 
dfp.columns = ['count', 'cached_quotes_found', 'channel_id', 'is_leg_subcomponent', 'lu_started', 'kind', 'agent_id', 'qr_status', 'quote_source', 'search_kind'] 

# function to classify each row as: none; cache; cache_and_lu; lu
def map_status(row):
    if row['cached_quotes_found']:
        if row['lu_started']:
            return 'cache_and_lu'
        else:
            return 'cache'
    else:
        if row['lu_started']:
            return 'lu'
        else:
            return 'none'   

dfp['summary_status'] = dfp.apply(map_status, axis=1)

# function to pivot a subset of the data by agent and status
def get_pivot(channel, search_kind):
    df_filtered = dfp[(dfp['channel_id'] == channel) & (dfp['search_kind'] == search_kind) & (dfp['is_leg_subcomponent'] == False)]
    df_piv = df_filtered.pivot_table(index='agent_id', columns='summary_status', values='count', aggfunc=np.sum)
    df_piv['total'] = df_piv.sum(axis=1)
    df_piv['cache_reuse_ratio'] = df_piv['cache'] / df_piv['total']
    return df_piv

# get a list of partners for return day view searches 
dfp_website_return_status = get_pivot('Website', 'RETURN') 

