import itertools
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

results = []
# START ----- VARIABLES
mandatory_columns = set(['Log_Price'])
max_features = 2 
min_features = 1 # >= len(mandatory_columns)
# all_results = []
columns_to_exclude = ['TOTAL_SALES', 'Year_Week', 'Row', 'LINE', 'DIV']
# END ------- VARIABLES

# i = 0
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
columns_to_exclude = set(map(lambda x: x+'_Flag', months) + columns_to_exclude)
columns = set(training_data1.columns) - columns_to_exclude
# this casts the string type flags into int
for column in filter(lambda (x, y): True if x.find('_Flag') > 0 and y == 'string' else False, training_data1.dtypes):
    training_data1 = training_data1.withColumn(column[0], training_data1[column[0]].cast("int"))

OR_OP = udf(lambda x, y: 1 if x + y > 0 else 0, IntegerType())

# combine months
for month_x, month_y in [months[i: i+2] for i in range(0, len(months), 2)]:
     training_data1 = training_data1.withColumn('%s_%s_Flag' % (month_x, month_y), OR_OP(training_data1[month_x+'_Flag'], training_data1[month_y+'_Flag']))

for n_features in range(len(mandatory_columns)+1, max_features+1):
    headers = map(lambda x: list(itertools.chain(x, mandatory_columns)), itertools.combinations(columns-mandatory_columns, n_features-1))
    for header in headers:
        print 'Iteration: %d Features : %s' % (n_features, header)
#         i = i + 1
        assembler_features = VectorAssembler(inputCols= header, outputCol="features")
        tmp = [assembler_features]
        pipeline = Pipeline(stages=tmp)
        train_DF0 = pipeline.fit(training_data1).transform(training_data1)
        val_DF0 = pipeline.fit(val_data1).transform(val_data1)
        trainingData = train_DF0.withColumn("label",col("TOTAL_UNITS").cast("integer")).select("label","features")
        testData = val_DF0.withColumn("label",col("TOTAL_UNITS").cast("integer")).select("label","features")
        lr = LinearRegression(maxIter=10, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True)
        lrModel = lr.fit(trainingData)
    #     lrModel = lr.fit(trainingData)
        trainingSummary = lrModel.summary
        for i in range(len(header)):
            results.append({'header': header[i],
                      'n_features': n_features,
                      'r_squared': trainingSummary.r2,
                      'coeff/value': lrModel.coefficients[i],
                      'p_value': trainingSummary.pValues[i],
                    })
        results.append({'header': 'intercept',
                  'n_features': n_features,
                  'r_squared': trainingSummary.r2,
                  'coeff/value': lrModel.intercept,
                  'p_value': trainingSummary.pValues[0],
                })
#             break
#             print results
results_df = pd.DataFrame(results)
results_df
# print i