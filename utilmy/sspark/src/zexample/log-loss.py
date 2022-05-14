from pyspark.sql.functions import udf
from pyspark.ml.feature import SQLTransformer

@udf('float')
def first_element(v): 
    return float(v[1])

log_lossTransformer = SQLTransformer(statement="""SELECT AVG(log(first_element(probability)) *  success  +  
                            log(1 - first_element(probability)) * (1 -success)) as log_loss 
                            FROM __THIS__""") 

model_output = model.fit(train_set) # first fit the model 
log_lossTransformer.transform(model_output.transform(test_set)).show()
