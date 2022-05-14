import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

class WaveTime(Transformer, HasInputCol, HasOutputCol):
    """Transform time field into sin/cos of seconds elapsed since midnight"""
    @keyword_only
    def __init__(self, inputCol=None, outputCol='wave_time'):
        super().__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]

        def split(dtime):
            seconds_from_midnight = dtime.hour * 60 * 60 + dtime.minute * 60 + dtime.second
            seconds_in_day = 24*60*60
            sin_time = np.sin(2 * np.pi * seconds_from_midnight/seconds_in_day)
            cos_time = np.cos(2 * np.pi * seconds_from_midnight/seconds_in_day)
            return Vectors.dense([sin_time, cos_time])

        return dataset.withColumn(out_col, udf(split, VectorUDT())(in_col))