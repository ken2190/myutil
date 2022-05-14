from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as sqlF
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import IntegerType


def add_label_churned(df):
    """Add the `churned` to indicate if the user churned"""
    
    # Identify the rows with a cancelled auth state and mark those 
    # with 1, then use a window function that groups
    # by users and puts the cancel event at the top (if any) so 
    # every row gets a one after that when we sum
    cancelled_udf = sqlF.udf(
        lambda x: 1 if x == 'Cancelled' else 0, IntegerType())
    current_window = Window.partitionBy('userId') \
        .orderBy(sqlF.desc('cancelled')) \
        .rangeBetween(Window.unboundedPreceding, 0)
    churned_df = df.withColumn('cancelled', cancelled_udf('auth')) \
        .withColumn("churned",
            sqlF.sum('cancelled').over(current_window))
    return churned_df.drop('cancelled')

def add_feature_number_sessions(df):
    """Returns a new dataframe with the `number_sessions` feature: total amount of sessions of the user"""
    counts_df = df.select('userId', 'sessionId') \
        .dropDuplicates() \
        .groupby('userId') \
        .count() \
        .withColumnRenamed('count', 'number_sessions')
    return df.join(counts_df, ['userId'])

def add_feature_number_sessions(df):
    """Add `number_sessions`: amount of sessions per user"""
    counts_df = df.select('userId', 'sessionId') \
        .dropDuplicates() \
        .groupby('userId') \
        .count() \
        .withColumnRenamed('count', 'number_sessions')
    return df.join(counts_df, ['userId'])

def add_feature_seconds_since_genesis(df):
    """Add `seconds_since_genesis`: seconds since first appearance"""
    current_window = Window.partitionBy('userId').orderBy('ts')
    genesis_df = df.withColumn(
        'seconds_since_genesis', 
        (sqlF.col('ts') - sqlF.first(sqlF.col('ts'))
                              .over(current_window)) / 1000.0)
    genesis_df = genesis_df.groupby('userId') \
        .max() \
        .withColumnRenamed(
            'max(seconds_since_genesis)', 'seconds_since_genesis'
        ).select('userId', 'seconds_since_genesis')
    return df.join(genesis_df, ['userId'])

def add_feature_avg_actions_per_session(df):
    """Add `avg_actions_per_session`: average actions per session"""
    current_window = Window.partitionBy('userId').orderBy('ts')
    avg_df = df.groupby('userId', 'sessionId') \
        .max() \
        .groupby('userId') \
        .avg() \
        .withColumnRenamed(
            'avg(max(itemInSession))', 'avg_actions_per_session'
        ).select('userId', 'avg_actions_per_session')
    return df.join(avg_df, ['userId'])

def add_feature_avg_seconds_per_session(df):
    """Add `avg_seconds_per_session`: average session duration"""
    current_window = Window.partitionBy(
        'userId', 'sessionId').orderBy('ts')
    avg_df = df.withColumn(
            'sessionLen', 
            (sqlF.col('ts') - sqlF.first(sqlF.col('ts')) \
                                  .over(current_window)) / 1000.0
        ).groupby('userId', 'sessionId') \
        .max() \
        .groupby('userId') \
        .avg() \
        .withColumnRenamed(
            'avg(max(sessionLen))', 'avg_seconds_per_session'
        ).select('userId', 'avg_seconds_per_session')
    return df.join(avg_df, ['userId'])

def load_df_for_ml(json_filepath):
    """Load json, then cleans/transform it for modeling"""
    df = spark.read.json(json_filepath)
    df_clean_v1 = df.filter('userId != ""')
    df_with_features = add_feature_number_sessions(df_clean_v1)
    df_with_features = add_feature_seconds_since_genesis(
        df_with_features)                   
    df_with_features = add_feature_avg_actions_per_session(
        df_with_features)                   
    df_with_features = add_feature_avg_seconds_per_session(
        df_with_features)                   
    df_with_features = add_label_churned(
        df_with_features)
    
    features = [
        'number_sessions', 'seconds_since_genesis', 
        'avg_actions_per_session', 'avg_seconds_per_session',
    ]
    return df_with_features.select(
        ['userId', 'churned'] + features).dropDuplicates()

def get_ml_pipeline(clf):
    """Constructs a pipeline to transform data before running clf"""
    features = [
        'number_sessions', 'seconds_since_genesis',
        'avg_actions_per_session', 'avg_seconds_per_session',
    ]
    assembler = VectorAssembler(
        inputCols=features, outputCol="features")
    return Pipeline(stages=[assembler, clf])

def eval_model(model, validation_df):
    """Runs a model against test set and prints performance stats"""
    results = model.transform(validation_df)    
    predictionAndLabels = results.rdd.map(
        lambda row: (float(row.prediction), float(row.label)))
    metrics = MulticlassMetrics(predictionAndLabels)
    print('Performance Stats')
    print(f'Accuracy: {metrics.accuracy:.4f}')
    print(f'Precision = {metrics.precision(1.0):.4f}')
    print(f'Recall = {metrics.recall(1.0):.4f}')
    print(f'F1 Score = {metrics.fMeasure(1.0):.4f}')

def build_cross_validator(numFolds=3):
    """Build CrossValidator for tuning a LogisticRegression model"""
    lr = LogisticRegression(standardization=True)
    pipeline = get_ml_pipeline(lr)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.0, 0.5]) \
        .addGrid(lr.aggregationDepth, [2, 4]) \
        .addGrid(lr.elasticNetParam, [0.0, 1.0]) \
        .addGrid(lr.maxIter, [10, 100]) \
        .build()
    evaluator = MulticlassClassificationEvaluator()    
    return CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=numFolds)