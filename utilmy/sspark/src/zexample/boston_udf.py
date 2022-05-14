from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

# setup the spark data frame as a table
boston_sp.createOrReplaceTempView("boston")

# add train/test label and expand the data set by 3x (each num trees parameter)
full_df = spark.sql("""
  select *
  from (
    select *, case when rand() < 0.8 then 1 else 0 end as training 
    from boston
  ) b
  cross join (
      select 11 as trees union all select 20 as trees union all select 50 as trees)
""")

schema = StructType([StructField('trees', LongType(), True),
                     StructField('r_squared', DoubleType(), True)])  

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def train_RF(boston_pd):
    trees = boston_pd['trees'].unique()[0]

    # get the train and test groups 
    boston_train = boston_pd[boston_pd['training'] == 1]
    boston_test = boston_pd[boston_pd['training'] == 0] 
        
    # create data and label groups 
    y_train = boston_train['target']
    X_train = boston_train.drop(['target'], axis=1)
    y_test = boston_test['target']
    X_test = boston_test.drop(['target'], axis=1)
   
    # train a classifier 
    rf= RFR(n_estimators = trees)
    model = rf.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)
    r = pearsonr(y_pred, y_test)
    
    # return the number of trees, and the R value 
    return pd.DataFrame({'trees': trees, 'r_squared': (r[0]**2)}, index=[0])
  
# use the Pandas UDF
results = full_df.groupby('trees').apply(train_RF)

# print the results 
print(results.take(3))
