from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(returnType=t.DoubleType())
def pd_predict(feature):
  temp = feature.values.tolist()
  return pd.Series(gs_rf.best_estimator_.predict_proba(temp)[:,1])

output_df = feature_df.withColumn('prob', pd_predict('features'))

display_cols = ['name', 'description', 'manufacturer', 'price']
sample_df = (
  output_df.filter(f.col('label').isNull())
  .select('edge.src', 'edge.dst', *[f.concat_ws('\nVS\n', 'src.' + c, 'dst.' + c).alias(c) for c in display_cols], 'overall_sim', 'prob')
  .sample(withReplacement=False, fraction=0.01, seed=42)
  .orderBy(f.col('prob').desc())
)

sample_df.write.mode('overwrite').csv("YOUR_STORAGE_PATH/candidate_pair_sample_v2.csv")