import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
 
 
@pandas_udf(returnType=DoubleType())
def predict_pandas_udf(*features):
    """ Executes the prediction using numpy arrays.
         
        Parameters
        ----------
        features : List[pd.Series]
            The features for the model, with each feature in it's
            owns pandas Series.
         
        Returns
        -------
        pd.Series
            The predictions.
    """
    # Need a multi-dimensional numpy array for sklearn models.
    X = pd.concat(features, axis=1).values
    # If model is somewhere in the driver we're good.
    y = model.predict(X)  # <- This is vectorized. Kachow.
    return pd.Series(y)