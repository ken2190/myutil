import os

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Adam
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType

from zoo.common.nncontext import *
from zoo.feature.image import *
from zoo.pipeline.nnframes import *

sc = init_nncontext("TransferLearningBlog")

model_path = '/Users/rafay/work/blog/analytics-zoo_resnet-50_imagenet_0.1.0.model'
train_path = '/Users/rafay/work/blog/data/train/*'
val_path = '/Users/rafay/work/blog/data/val/*'