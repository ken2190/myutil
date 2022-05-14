import pyspark.sql.functions as F
import pyspark.sql.types as T


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT

def avg_vectors(bert_vectors):
  length = len(bert_vectors[0]["embeddings"])
  avg_vec = [0] * length
  for vec in bert_vectors:
    for i, x in enumerate(vec["embeddings"]):
      avg_vec[i] += x
    avg_vec[i] = avg_vec[i] / length
  return avg_vec


#create a udf
avg_vectors_udf = F.udf(avg_vectors, T.ArrayType(T.DoubleType()))
df_doc_vec = df_bert.withColumn("doc_vector", avg_vectors_udf(F.col("embeddings")))
display(df_doc_vec)



def dense_vector(vec):
	return Vectors.dense(vec)

dense_vector_udf = F.udf(dense_vector, VectorUDT())
training = df_doc_vec.withColumn("features", dense_vector_udf(F.col("doc_vector")))


lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrParisModel = lr.fit(training)
print("Coefficients: " + str(lrParisModel.coefficients))
print("Intercept: " + str(lrParisModel.intercept))