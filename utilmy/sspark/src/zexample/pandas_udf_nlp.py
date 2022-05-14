import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType
import spacy

#...
# nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_sm')
#...

# Use pandas_udf to define a Pandas UDF
@pandas_udf('array<double>', PandasUDFType.SCALAR)
# The input is a pandas.Series with strings. The output is a pandas.Series of arrays of double.
def pandas_nlp(s):
    return s.fillna("_NO_₦Ӑ_").replace('', '_NO_ӖӍΡṬΫ_').transform(lambda x: (nlp(x).vector.tolist()))
