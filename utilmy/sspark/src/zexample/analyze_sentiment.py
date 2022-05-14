import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row, SQLContext, DataFrameReader, functions
from pyspark.sql.functions import udf, create_map, collect_list
from pyspark.sql.types import ArrayType, StringType, DoubleType
import json
import re
from pyspark.ml.feature import HashingTF, IDF, Normalizer, ElementwiseProduct, Word2Vec, StopWordsRemover, Tokenizer, RegexTokenizer
from pyspark.ml.linalg import Vectors, DenseVector, VectorUDT
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans
import pandas as pd
import numpy as np


def CleanReviews(df):

	tokenizer = RegexTokenizer(inputCol='reviewText', outputCol='tokenized', pattern='[^a-zA-Z]+')
	tokenized = tokenizer.transform(df)
	tokenized.show(10)


	remover = StopWordsRemover(inputCol='tokenized', outputCol='filtered')
	removed = remover.transform(tokenized)
	removed.show(10)

	removed = removed.drop('reviewText')
	removed = removed.drop('tokenized')
	removed = removed.withColumnRenamed('filtered', 'reviewText')
	
	
	return removed

def CreateTFIDF(df):
	# compute term frequency for each element
	hashingTF = HashingTF(numFeatures=50000, inputCol='reviewText', outputCol='tf')

	# compute tf idf
	idf = IDF(minDocFreq=10, inputCol='tf', outputCol='idf')

	normalizer = Normalizer(inputCol='idf', outputCol='featurevec_norm')
	lr = LinearRegression(maxIter=10, featuresCol='featurevec_norm', labelCol='overall')
	lreval = RegressionEvaluator(labelCol='overall')

	pipeline = Pipeline(stages=[hashingTF, idf, normalizer, lr])
	paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
	crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=lreval, numFolds=5)
	cvmodel = crossval.fit(df)
	pred = cvmodel.transform(df)
	pred.show(10)
	print lreval.evaluate(pred)

	return

def ApplyWord2Vec(df):


	word2vec = Word2Vec(inputCol='reviewText', outputCol='featurevec', vectorSize=5)
	lr = LinearRegression(maxIter=10, featuresCol='featurevec', labelCol='overall')
	lreval = RegressionEvaluator(labelCol='overall')

	pipeline = Pipeline(stages=[word2vec, lr])
	paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
	crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=lreval, numFolds=5)
	cvmodel = crossval.fit(df)
	pred = cvmodel.transform(df)
	pred.show(10)
	print lreval.evaluate(pred)

def MakeDict(model):
	mymap = model.select(create_map('word', 'cluster').alias('map'))
	mylist = mymap.select(collect_list(mymap.map).alias('dict')).head()['dict']
	d = {}
	for elem in mylist:
		for key in elem:
			d[key] = elem[key]

	return d

def ClustVecMaker(s, d, nClust):

	sv = {}

	for x in s:
		try:
			cid = d[x]
			sv[cid] = 1
			#print 'success', x

		except Exception, e:
			#print 'failure', x
			continue
	

	return Vectors.sparse(nClust, sv)


def ApplyClustering(df):

	word2vec = Word2Vec(inputCol='reviewText', vectorSize=5, minCount=100)
	w2vModel = word2vec.fit(df)

	wordVectorsDF = w2vModel.getVectors()
	nClust = wordVectorsDF.count() / 20

	kmeans = KMeans(k=nClust, featuresCol='vector')
	modelK = kmeans.fit(wordVectorsDF)
	clustersDF = modelK.transform(wordVectorsDF).select('word', 'prediction').withColumnRenamed('prediction', 'cluster').orderBy('cluster')
	clustersDF.show(100)

	d = MakeDict(clustersDF)
	#print d
	#return 
	

	clustvecmaker = udf(lambda s: ClustVecMaker(s, d, nClust), VectorUDT())
	clustvecs = df.withColumn('featurevec', clustvecmaker(df.reviewText))

	normalizer = Normalizer(inputCol='featurevec', outputCol='featurevec_norm')
	lr = LinearRegression(maxIter=10, featuresCol='featurevec_norm', labelCol='overall')
	lreval = RegressionEvaluator(labelCol='overall')

	pipeline = Pipeline(stages=[normalizer, lr])
	paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.solver, ['l-bfgs']).build()
	crossval = CrossValidator(estimator=pipeline, evaluator=lreval, estimatorParamMaps=paramGrid, numFolds=5)

	cvmodel = crossval.fit(clustvecs)
	pred = cvmodel.transform(clustvecs)
	pred.show(10)
	print lreval.evaluate(pred)

def main(f):
	df = spark.read.format('json'). load(infile).repartition(32, 'unixReviewTime')
	df = df.select(df.reviewText, df.overall)
	#print df.dtypes

	df = CleanReviews(df)
	df.cache()
	df.show(10)

	#CreateTFIDF(df)
	#ApplyWord2Vec(df)
	ApplyClustering(df)

	return


spark = SparkSession.builder.appName("Simple App").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")


infile = sys.argv[1]

main(infile)


