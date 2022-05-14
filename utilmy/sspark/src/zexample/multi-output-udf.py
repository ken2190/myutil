from pyspark.sql import Row
import pyspark.sql.functions as F

def append_payer_spend(context_ts, collected_col):
    if len(collected_col) == 1:
        if collected_col[0] == Row(None,None):
            return Row('is_payer', 'spend')(0.0, 0.0)
    
    collected_col = sorted(collected_col, key=lambda x: x.txTimestamp, reverse=False)
    is_payer = 0.0
    total_spend = 0.0
    for entry in collected_col:
        diff = (entry.txTimestamp - context_ts).days
        if diff >= 0 and diff < 7:
            is_payer = 1.0
            total_spend += entry.receiptUsdAmount
    return Row('is_payer', 'spend')(is_payer, total_spend)

# struct to store multiple values
schema_added = StructType([
    StructField("is_payer", FloatType(), False),
    StructField("spend", FloatType(), False)])

append_payer_spend_udf = F.udf(append_payer_spend, schema_added)

new_df = df_likely_payer.withColumn("output", \
                                    append_payer_spend_udf(df_likely_payer['ts'], df_likely_payer['collected_col']))\
        .select(*(df_likely_payer.columns), 'output.*').drop('collected_col')
