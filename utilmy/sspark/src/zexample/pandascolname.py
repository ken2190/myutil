def new_col_from_function_using_pdf_column_inputs(
        df: pd.DataFrame,
        f: Callable,
        new_series_name: str = None,
        constant_argument_name_value_dict: dict = {},
):
    """Make a Series from a function, auto-pulling arguments from a DataFrame
    
    Arguments to the python function are automatically pulled in from the
    pandas dataframe where the python function's argument name matches the
    pandas dataframe's Series name.
    
    Inputs are determined with the following priority: if the argument name in f

    1) is a key in argument_name_value_dict, the corresponding value of the
        dict is used (same value used for every row of the spark data frame)

    2) matches a column name in df, that column is used

    3) has a default specified in f, that value is used (same value used
        for every row of the spark dataframe)

    4) doesn't meet any of the above criteria, a ValueError is raised

    # Args:
    * sdf (pyspark.sql.dataframe.DataFrame): DataFrame to add column to
    * f (function): a python function use to make the new column
    * new_col_name (string): the column name for the column this function adds to sdf
    * returnType (pyspark.sql.types): the type returned by f
        defaults to pyspark.sql.types.FloatType(). Other common types include
        IntegerType, StringType, BooleanType. Search google for more.
    * constant_argument_name_value_dict (dict): a dictionary of 
          argument name: argument values
        to supply to f. These get first priority to be passed as inputs to f.
    * functionType: an enum value in :class:`pyspark.sql.functions.PandasUDFType`
        see pandas_udf() documentation

    # Returns:
    * pyspark.sql.dataframe.DataFrame: sdf with another column named new_col_name
    """

    # if no name given, use name of function
    if new_series_name is None:
        new_series_name = f.__name__

    logger.debug(
        f"Adding column {new_series_name} to spark DataFrame using python function {f.__name__}"
    )

    # get the arguments of the python function
    argspec = inspect.getfullargspec(f)
    python_function_argument_names = argspec.args

    # get the default values and the index of the first default
    python_function_defaults = argspec.defaults
    num_args = len(python_function_argument_names
                   ) if python_function_argument_names else 0
    num_default_args = len(
        python_function_defaults) if python_function_defaults else 0
    first_default_idx = num_args - num_default_args

    # list of inputs for UDF to build up
    inputs = []

    # list of df columns to confirm they exist if value not provided for arg
    df_columns = set(df.columns)

    # loop over args
    for idx, argument in enumerate(python_function_argument_names):
        if argument in constant_argument_name_value_dict:
            # if in argument_name_value_dict, use specified value
            inputs.append(
                constant_argument_name_value_dict[argument])
            logger.debug(
                f"Using dictionary supplied value {constant_argument_name_value_dict[argument]} for {argument}"
            )
        else:
            # otherwise if in df_columns, use the column's value
            if argument in df_columns:
                inputs.append(df[argument])
                logger.debug(f"Using input pandas dataframe column for {argument}")
            else:
                # otherwise if a default is specified in the function, use it
                if idx >= first_default_idx:
                    inputs.append(
                            python_function_defaults[idx - first_default_idx])
                    logger.debug(
                        f"Using function default value {python_function_defaults[idx - first_default_idx]} for {argument}"
                    )
                else:
                    # otherwise the user hasn't specified a required argument
                    raise ValueError(
                        f"{argument} not specified and not in df input")
    return f(*inputs).rename(new_series_name)