import pandas as pd
from typing import Iterator
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, FloatType, StringType, StructField
import xgboost as xgb

def inference_func_factory(model):
    def inference_func(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for row in iterator:
            yield predict(row, model)
    return inference_func


output_schema_columns = [StructField('moviegoer_id',
                                    StringType(),
                                    False),
                        StructField('visit_probability',
                                    FloatType(),
                                    False)]
output_schema = StructType(output_schema_columns)

model = xgb.XGBClassifier()
model.load_model(model_path)

predictions = (inference_dataset
            .mapInPandas(inference_func_factory(model),
                            schema=output_schema))