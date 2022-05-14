from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql import Window as w

from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, CountVectorizer, StopWordsRemover, NGram, Normalizer, VectorAssembler, Word2Vec, Word2VecModel, PCA
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.linalg import VectorUDT, Vectors
import tensorflow_hub as hub

def tokenize(df, string_cols):
  output = df
  for c in string_cols:
    output = output.withColumn('temp', f.coalesce(f.col(c), f.lit('')))
    tokenizer = RegexTokenizer(inputCol='temp', outputCol=c+"_tokens", pattern = "\\W")
    remover = StopWordsRemover(inputCol=c+"_tokens", outputCol=c+"_swRemoved")
    output = tokenizer.transform(output)
    output = remover.transform(output)\
      .drop('temp', c+"_tokens")
    
  return output

def top_kw_from_tfidf(vocab, n=3):
  @udf(returnType=t.ArrayType(t.StringType()))
  def _(arr):
    inds = arr.indices
    vals = arr.values
    top_inds = vals.argsort()[-n:][::-1]
    top_keys = inds[top_inds]
    output = []

    for k in top_keys:
      kw = vocab.value[k]
      output.append(kw)

    return output
  return _

def tfidf_top_tokens(df, token_cols, min_freq=1):
  output = df
  for c in token_cols:
    pre = c
    cv = CountVectorizer(inputCol=pre, outputCol=pre+'_rawFeatures', minDF=min_freq)
    idf = IDF(inputCol=pre+"_rawFeatures", outputCol=pre+"_features", minDocFreq=min_freq)
    normalizer = Normalizer(p=2.0, inputCol=pre+"_features", outputCol=pre+'_tfidf')
    stages = [cv, idf, normalizer]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(output)
    output = model.transform(output)\
      .drop(pre+'_rawFeatures', pre+'_features')
    
    cvModel = model.stages[0]
    vocab = spark.sparkContext.broadcast(cvModel.vocabulary)
    output = output.withColumn(pre+'_top_tokens', top_kw_from_tfidf(vocab, n=5)(f.col(pre+"_tfidf")))
  
  return output
      

# magic function to load model only once per executor
MODEL = None
def get_model_magic():
  global MODEL
  if MODEL is None:
      MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  return MODEL

@udf(returnType=VectorUDT())
def encode_sentence(x):
  model = get_model_magic()
  emb = model([x]).numpy()[0]
  return Vectors.dense(emb)
  
blocking_df = tokenize(processed_df, ['name', 'description', 'manufacturer'])
blocking_df = tfidf_top_tokens(blocking_df, [c+'_swRemoved' for c in ['name', 'description', 'manufacturer']])
blocking_df = blocking_df.withColumn('name_encoding', encode_sentence(f.coalesce(f.col('name'), f.lit(''))))\
  .withColumn('description_encoding', encode_sentence(f.coalesce(f.col('description'), f.lit(''))))\
  .withColumn('blocking_keys',
                f.array_union(f.col('name_swRemoved_top_tokens'), f.array_union(f.col('description_swRemoved_top_tokens'), f.col('manufacturer_swRemoved_top_tokens')))
             )\
  .withColumn('uid', f.concat_ws('|', 'source', 'source_id'))