from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA

vectorizer = CountVectorizer(inputCol= "stemmed_rm", outputCol="rawFeatures")
# 2.5. IDf
idf = IDF(inputCol="rawFeatures", outputCol="features")

# 3. LDA model
lda = LDA(k=n_topics, seed=seedNum, optimizer="em", maxIter=maxIter)

pipeline = Pipeline(stages=[vectorizer, idf, lda])

pipeline_model = pipeline.fit(df_train)
pipeline_model.write().overwrite().save(s3_bucket + pipelinePath + 
    "ntopics_" + str(n_topics) + "_maxIter_" + str(maxIter))