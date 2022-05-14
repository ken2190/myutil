import featuretools as ft
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_feature_generation(pandasInputDF):
    
    # create Entity Set representation 
    es = ft.EntitySet(id="events")
    es = es.entity_from_dataframe(entity_id="events", dataframe=pandasInputDF)
    es = es.normalize_entity(base_entity_id="events", new_entity_id="users", index="user_id")
    
    # apply the feature calculation and return the result 
    return ft.calculate_feature_matrix(saved_features, es)
  
sparkFeatureDF = sparkInputDF.groupby('user_group').apply(apply_feature_generation)