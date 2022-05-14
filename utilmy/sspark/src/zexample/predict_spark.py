"""Main Entrypoint to submit to the Spark Cluster"""

import os
from typing import Tuple

import pandas as pd
import torch
from data_components.io.files.s3 import Client
from pyspark.sql import SparkSession
from pyspark.sql.functions import PandasUDFType, col, pandas_udf
from pyspark.sql.types import FloatType, StringType
from scipy.special import expit
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from . import TokensDataset

BATCH_SIZE: int = 250


def run(spark, sc, src, dst, model_path):

    global BC_MODEL
    global BC_TOKENIZER

    tokenizer, model_loaded = load_model(model_path)

    # NOTE: Distribute de models into the cluster
    BC_TOKENIZER = sc.broadcast(tokenizer)
    BC_MODEL = sc.broadcast(model_loaded)

    # NOTE: Define udfs
    input_id_udf = pandas_udf(spark_input_id, returnType=StringType())
    attention_mask_udf = pandas_udf(spark_attention_mask, returnType=StringType())
    predict_udf = pandas_udf(FloatType(), PandasUDFType.SCALAR)(predict_batch)

    df = spark.read.format('parquet').load(src, compression="snappy")
    df = (
        df.select(["id", "an", "publication_date", "modification_date", "body_masked"])
        .withColumn('input_ids', input_id_udf(df.body_masked))
        .withColumn('attention_mask', attention_mask_udf(df.body_masked))
        .select(
            col("id"),
            col('an'),
            col("publication_date"),
            col("modification_date"),
            col('body_masked'),
            predict_udf(col('input_ids'), col("attention_mask")).alias("prediction")
        )
    )

    df.write.mode("overwrite").format('parquet').save(dst)


def download_model(path, local_path):
    client = Client()
    client.download(remote=path, local=local_path)


def get_model():
    """Gets the broadcasted model."""
    global BC_MODEL
    return BC_MODEL.value


def get_tokenizer():
    """Gets the broadcasted tokenizer."""
    global BC_TOKENIZER
    return BC_TOKENIZER.value


def spark_input_id(t):
    def input_id(x):
        tokenizer = get_tokenizer()
        return str(tokenizer(str(x), padding=True, truncation=True)['input_ids'])
    return t.apply(lambda x: input_id(x))


def spark_attention_mask(t):
    def attention_mask(x):
        tokenizer = get_tokenizer()
        return str(tokenizer(str(x), padding=True, truncation=True)['attention_mask'])
    return t.apply(lambda x: attention_mask(x))


def predict_batch(input_ids, attention_mask):

    model = get_model()
    final_tokens = TokensDataset(input_ids, attention_mask)
    loader = torch.utils.data.DataLoader(
        final_tokens,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            tensor_input_ids = torch.as_tensor(batch.input_ids, dtype=torch.long).to('cpu')
            tensor_attention_mask = torch.as_tensor(batch.attention_mask, dtype=torch.long).to('cpu')
            outputs = model(input_ids=tensor_input_ids, attention_mask=tensor_attention_mask)
            all_predictions.extend(outputs.logits.flatten().tolist())

    return pd.Series(all_predictions)


def load_model(model_path):
    """Return the model and tokenizer"""
    detoke = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, return_dict=True)

    return detoke, model


if __name__ == "__main__":
    spark = SparkSession.builder.appName("controversies_emr").getOrCreate()
    sc = spark.sparkContext
    src = os.getenv('SRC')
    dst = os.getenv('DST')
    model_path = os.getenv('MODEL_PATH')
    run(spark, sc, src, dst, model_path)
