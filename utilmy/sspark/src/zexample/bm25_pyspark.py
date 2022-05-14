import numpy as np

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer

udf_template = """

def bm25(query, tf, idf):
    
    mean_dl = {}
    k = {}
    b = {}
    
    idf_values, tf_indices, tf_values, query_indices = idf.values, tf.indices, tf.values, query.indices
    freq_nidx = tf_indices[-1]+1
    freq_indices = np.concatenate((tf_indices, np.array([freq_nidx])))
    term_frequencies = np.concatenate((tf_values, np.array([0.0])))
    
    #get idf vector
    idf_ = idf_values
    
    #get term frequencies
    intersect = np.intersect1d(query_indices, freq_indices)
    idx = np.where(np.isin(query_indices, intersect), query_indices, freq_nidx)
    freq_idx = np.searchsorted(freq_indices, idx)
    tf_ = term_frequencies[freq_idx].reshape(-1)
    
    #get doc length
    dl_ = tf_values.sum()
    ntf_ = tf_ / dl_
    tf_score = ntf_.sum()
    tfidf_score = np.dot(ntf_, idf_)
    
    n_term = k * (1 - b + b * dl_/mean_dl)
    
    bm25 = np.dot(idf_, (tf_ * (k + 1)) / ((tf_) + n_term))
    
    return T.Row('ntf', 'tfidf', 'bm25')(float(tf_score), float(tfidf_score), float(bm25))
    
schema = T.StructType([
    T.StructField("ntf", T.FloatType(), False),
    T.StructField("tfidf", T.FloatType(), False),
    T.StructField("bm25", T.FloatType(), False)])
    
self.udf = F.udf(bm25, returnType=schema)

"""

class BM25_Model(object):
    """
    Computes BM25 score.
    """
    def __init__(self, k=1.2, b=.75):
        self.k = k
        self.b = b
        self.tok = Tokenizer(inputCol='__input', outputCol='__tokens')
        self.vec = CountVectorizer(inputCol='__tokens', outputCol='__counts')
        self.idf = IDF(inputCol='__counts', outputCol='__idf')
        self.train_col = None
        self.udf = None
        
    def fit(self, df, train_col):
        """
        Does fitting on input df.
        """
        self.train_col = train_col
        df_ = self.tok.transform(df.withColumnRenamed(train_col, '__input'))
        mean_dl = df_.select(F.mean(F.size(F.col('__tokens')))).collect()[0][0]
        self.vec = self.vec.fit(df_)
        df_ = self.vec.transform(df_)
        self.idf = self.idf.fit(df_)
        #this will reset value of self.udf to be a working udf function.
        exec(udf_template.format(mean_dl, self.k, self.b))
        
    def transform(self, df, score_col, bm25_output_name='bm25', ntf_output_name=None, tfidf_output_name=None):
        """
        Computes BM25 score, 
            along with normalized term frequency (ntf) and tfidf.
            These two additional scores come "for free" with bm25
            but are only returned optionally.
        """
        columns = df.columns
        df_ = self.tok.transform(df.withColumnRenamed(score_col, '__input'))
        df_ = self.vec.transform(df_)
        df_ = self.idf.transform(df_)
        df_ = (df_.withColumnRenamed('__counts', '__query_counts')
               .withColumnRenamed('__input', score_col)
              ).select(columns + [score_col, '__query_counts', '__idf'])
        df_ = self.tok.transform(df_.withColumnRenamed(self.train_col, '__input'))
        df_ = self.vec.transform(df_)
        df_ = df_.withColumnRenamed('__counts', '__item_counts')
        df_ = df_.withColumn('bm25', self.udf(F.col('__query_counts'), F.col('__item_counts'), F.col('__idf')))
        df_ = df_.withColumnRenamed('__input', self.train_col)
        computed_values = df_.withColumn('more', F.explode(F.array(F.col('bm25')))).select(columns + ['bm25.*'])
        
        #this is logic for naming output column(s)
        final_selection = columns
        if bm25_output_name is not None:
            computed_values = computed_values.withColumnRenamed('bm25', bm25_output_name)
            final_selection.append(bm25_output_name)
        if ntf_output_name is not None:
            computed_values = computed_values.withColumnRenamed('ntf', ntf_output_name)
            final_selection.append(ntf_output_name)
        if tfidf_output_name is not None:
            computed_values = computed_values.withColumnRenamed('tfidf', tfidf_output_name)
            final_selection.append(tfidf_output_name)
        
        return computed_values.select(final_selection)