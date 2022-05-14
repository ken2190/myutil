from pyspark.sql.functions import pandas_udf, PandasUDFType


@pandas_udf('string', PandasUDFType.SCALAR)
def get_zip_pdf_b(lat_series, lng_series):
    pdf = brd_pdf.value
    zip_series= []
    for k in range(len(lat_series)):
        lat = lat_series[k]
        lng = lng_series[k]
        try:
            out = pdf[(pdf['bounds_north']>=lat) &
                      (pdf['bounds_south']<=lat) &
                      (pdf['bounds_west']<=lng) &
                      (pdf['bounds_east']>=lng) ]
            dist = [None]*len(out)
            for i in range(len(out)):
                dist[i] = (out['lat'].iloc[i]-lat)**2 + (out['lng'].iloc[i]-lng)**2
            zip = out['zipcode'].iloc[dist.index(min(dist))]
        except:
            zip = 'bad'
        zip_series.append(zip)
    return pd.Series(zip_series)