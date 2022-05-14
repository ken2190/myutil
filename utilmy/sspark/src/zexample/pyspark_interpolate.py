# define interpolation function
def interpol(x, x_prev, x_next, y_prev, y_next, y):
    if x_prev == x_next:
        return y
    else:
        m = (y_next-y_prev)/(x_next-x_prev)
        y_interpol = y_prev + m * (x - x_prev)
        return y_interpol

# convert function to udf
interpol_udf = func.udf(interpol, FloatType())   
    
# add interpolated columns to dataframe and clean up
df_filled = df_filled.withColumn('readvalue_interpol', interpol_udf('readtime', 'readtime_ff', 'readtime_bf', 'readvalue_ff', 'readvalue_bf', 'readvalue'))\
                     .drop('readtime_existent', 'readtime_ff', 'readtime_bf')\
                     .withColumnRenamed('reads_all', 'readvalue')\
                     .withColumn('readtime', func.from_unixtime(col('readtime')))