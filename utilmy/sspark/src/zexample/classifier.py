from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import OneVsRest, OneVsRestModel
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql.functions import (
    udf,
    lit,
    monotonically_increasing_id,
    collect_list,
    desc
)
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoderEstimator,
    VectorAssembler
)
from pyspark.sql import DataFrame
import functools

__all__ = ['MultiLabelClassifier']

class MultiLabelClassifier:
    """
    Multi-label classifier implementation.
    :param mdl_instance: an instance of the model to be used
    :type mdl_instance: pyspark.ml.classification
    Example
    -------
        >>> mlc = MultiLabelClassifier(LogisticRegression())
        >>> mlc.fit(df)
        >>> preds = mlc.transform(df)
    """

    def __init__(self, mdl_instance):
        self.ovr = OneVsRest(classifier=mdl_instance)
        self.fitted_mlc = None

    def fit(self, df):
        """
        Fit multi-label classifier
        :param df: dataframe with `features` and `label` columns
        :type df: pyspark.sql.DataFrame
        """
        pipe = Pipeline(stages=[self.ovr])
        self.fitted_mlc = pipe.fit(df)
        
    def save(self, path):
        """
        Save fitted multi-label classifier model
        :param path: path to save model to
        :type path: string
        """
        self.fitted_mlc.save()
        
    def __repr__(self):
        return 'MultiLabel({})'.format(self.ovr)