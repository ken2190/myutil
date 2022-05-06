def hit_rate_k(pred_df, actual_col="item_id", pred_col="predictions"):
    """.
    Doc::
            
            pred_df: DataFrame containing what a user actually interacted with and a predicted list
            actual_col: Column that has what the user actually engaged in
            pred_col: Column name that has the predictions in list form
            :return: Fractional hit rate for any predictions in `pred_col`
    """

    pred_df["hit"] = [
        actual in pred for pred, actual in pred_df[[pred_col, actual_col]].values
    ]

    return pred_df["hit"].mean()
