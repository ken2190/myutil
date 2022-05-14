# broadcast the row of interest and ordered feature names
ROW_OF_INTEREST_BROADCAST = spark.sparkContext.broadcast(
    row_of_interest[features_col]
)
ORDERED_FEATURE_NAMES = spark.sparkContext.broadcast(feature_names)

# set up the udf - x-j and x+j need to be calculated for every row
def calculate_x(
        feature_j, z_features, curr_feature_perm
):
    """
    The instance  x+j is the instance of interest,
    but all values in the order before feature j are
    replaced by feature values from the sample z
    The instance  x−j is the same as  x+j, but in addition
    has feature j replaced by the value for feature j from the sample z
    """
    x_interest = ROW_OF_INTEREST_BROADCAST.value
    ordered_features = ORDERED_FEATURE_NAMES.value
    x_minus_j = list(z_features).copy()
    x_plus_j = list(z_features).copy()
    f_i = curr_feature_perm.index(feature_j)
    after_j = False
    for f in curr_feature_perm[f_i:]:
        # replace z feature values with x of interest feature values
        # iterate features in current permutation until one before j
        # x-j = [z1, z2, ... zj-1, xj, xj+1, ..., xN]
        # we already have zs because we go row by row with the udf,
        # so replace z_features with x of interest
        f_index = ordered_features.index(f)
        new_value = x_interest[f_index]
        x_plus_j[f_index] = new_value
        if after_j:
            x_minus_j[f_index] = new_value
        after_j = True

    # minus must be first because of lag
    return Vectors.dense(x_minus_j), Vectors.dense(x_plus_j)

udf_calculate_x = F.udf(calculate_x, T.ArrayType(VectorUDT()))
