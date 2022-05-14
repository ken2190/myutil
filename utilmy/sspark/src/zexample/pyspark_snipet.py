def entropy(df, key, grp_cols, value_col):
    return df.groupby(grp_cols)\
             .agg(fn.sum(value_col).alias('sum_' + value_col))\
             .withColumn('total_' + value_col, fn.sum('sum_' + value_col).over(Window.partitionBy(key)))\
             .withColumn('frac', fn.col('sum_' + value_col) / fn.col('total_' + value_col))\
             .groupby(key)\
             .agg((-(fn.sum(fn.col('frac') * fn.log2('frac')))).alias('entropy_' + value_col))


def size_by_size_barchart(values_dict, columns, ax, xlabels, prefix_legend):
    gap = .8 / len(columns)
    
    for i, col in enumerate(columns):
        X = np.arange(len(values_dict[col]))
        ax.bar(X + i * gap, values_dict[col], width=gap)
    ax.set_xticklabels([''] + xlabels)
    ax.legend([prefix_legend + col for col in columns])
 

bins = [[-float("inf"), 0, 1.8, 3, 5, 6, float("inf")]]

def bucketize(df, columns, bins):
    from pyspark.ml.feature import Bucketizer
    buckets = []
    for i, col in enumerate(columns):
        bucketizer = Bucketizer(splits=bins[i], inputCol=col, outputCol=col + '_bucketized')
        df_buck = bucketizer.setHandleInvalid("keep").transform(df)
    return df_buck

df_label_train = bucketize(df_label_train, ['label'], bins)


def recode(col_name, map_dict, default=None):
    from itertools import chain
    from pyspark.sql.column import Column
    if not isinstance(fn.col, Column):
        col_name = fn.col(col_name)
    mapping_expr = fn.create_map([fn.lit(x) for x in chain(*map_dict.items())])
    if default is None:
        return  mapping_expr.getItem(col_name)
    else:
        return fn.when(~fn.isnull(mapping_expr.getItem(col_name)), mapping_expr.getItem(col_name)).otherwise(default)
    
    
# PSI: https://mwburke.github.io/2018/04/29/population-stability-index.html
# PSI < 0.1: no significant population change
# PSI < 0.2: moderate population change
# PSI >= 0.2: significant population change

from pyspark.ml.feature import QuantileDiscretizer, Bucketizer

def bin_bucketize(df, columns, bins):
    for i, col in enumerate(columns):
        bucketizer = Bucketizer(splits=bins[i], inputCol=col, outputCol=col + '_bucketized')
        df = bucketizer.setHandleInvalid("keep").transform(df)
    return df

def quantile_bucketize(df, columns, n=10):
    bucketizers = {}
    for i, col in enumerate(columns):
        bucketizer = QuantileDiscretizer(numBuckets=n, inputCol=col, outputCol=col + '_bucketized').fit(df)
        bucketizers[col] = bucketizer
        df = bucketizer.transform(df)
    return df, bucketizers


def quantile_bucketize_transform(df, columns, models, n=10):
    for i, col in enumerate(columns):
        bucketizer = models[col]
        df = bucketizer.transform(df)
    return df


def psi(df_expected, df_actual, columns):
    df_expected, models = quantile_bucketize(df_expected, columns)
    df_actual = quantile_bucketize_transform(df_actual, columns, models)
    bucket_columns = [col + '_bucketized' for col in columns]
    results = {}
    for col, buck_col in zip(columns, bucket_columns):
        df_col_grp_exp = df_expected.groupby(buck_col)\
                                    .count()\
                                    .withColumn('e_perc', 
                                                fn.col('count') / fn.sum('count').over(Window.partitionBy()))\
                                    .drop('count')
        
        df_col_grp_act = df_actual.groupby(buck_col)\
                                  .count()\
                                  .withColumn('a_perc', 
                                              fn.col('count') / fn.sum('count').over(Window.partitionBy()))\
                                  .drop('count')
        
        df_col_grp = df_col_grp_exp.join(df_col_grp_act, buck_col, 'outer').fillna(0.0001)\
                           .withColumn('psi', 
                                       (fn.col('e_perc') - fn.col('a_perc')) * (fn.log('e_perc') - fn.log('a_perc')))
        
        results[col] = df_col_grp.select('psi').collect()[0][0]
    return results

quantiles = fn.expr('percentile_approx(degree, array(0.25, 0.5, 0.75))')


# FEAT
def most_visited_location(df, k=2, loc_type='pincode'):
    # return the most frequently visited location of a user by month
    # Args:
        # k: return upto k-th most visited locations
        # loc_type: pincode/lacci
    return df.withColumn('month', month_udf('date'))\
             .groupby('sid', 'month', loc_type)\
             .agg(fn.count('*').alias('n_hours'),
                  fn.sum('num_interactions').alias('n_events'))\
             .withColumn('rank', fn.dense_rank().over(Window.partitionBy('sid', 'month')\
                                 .orderBy(fn.desc('n_hours'))))\
             .filter('rank <= {}'.format(k))

def prob_at_kth_location(df, df_freq_loc, k=2, loc_type='pincode'):
    # return the probability of finding a user at upto her k-th most frequently visited location by month
    return df.withColumn('month', month_udf('date'))\
             .groupby('sid', 'month', loc_type)\
             .agg(fn.count('*').alias('n_hours'),
                  fn.sum('num_interactions').alias('n_events'))\
             .withColumn('total_hours', fn.sum('n_hours').over(Window.partitionBy('sid', 'month')))\
             .withColumn('total_events', fn.sum('n_events').over(Window.partitionBy('sid', 'month')))\
             .withColumn('time_prob', col('n_hours') / col('total_hours'))\
             .withColumn('event_prob', col('n_hours') / col('total_events'))\
             .join(df_freq_loc, ['sid', 'month', loc_type])\
             .select('sid', 'month', 'time_prob', 'event_prob', 'rank')\
             .groupby('sid', 'month')\
             .pivot('rank', [1, 2])\
             .agg(fn.first('time_prob').alias('time_prob'),
                  fn.first('event_prob').alias('event_prob'))\
             .fillna(0)\
             .withColumn('diff_time', (col('1_time_prob') - col('2_time_prob')) / col('1_time_prob'))\
             .withColumn('diff_event', (col('1_event_prob') - col('2_event_prob')) / col('1_event_prob'))

def position_1st_2nd_position(df_freq_loc, df_pincode_info):
    return df_freq_loc.join(df_pincode_info, 'pincode')\
                      .groupby('sid', 'month')\
                      .pivot('rank', [1, 2])\
                      .agg(fn.first('lat').alias('lat'),
                           fn.first('lng').alias('lng'))\
                      .na.drop()\
                      .withColumn('dist_1st_2nd', haversine_udf('1_lat', '1_lng', '2_lat', '2_lng'))