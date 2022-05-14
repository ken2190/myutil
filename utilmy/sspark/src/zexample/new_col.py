def new_col_from_python_function_using_sdf_column_inputs(
        sdf: DataFrame,
        f: Callable,
        new_col_name: str = None,
        returnType=FloatType(),
        constant_argument_name_value_dict: dict = {},
        functionType=PandasUDFType.SCALAR,
) -> DataFrame:
    """Add a new column to a spark dataframe using a python function
    Arguments to the python function are automatically pulled in from the
    spark dataframe where the python function's argument name matches the
    spark dataframe's column name.

    First, f is converted to a UDF. Second, the function determines the inputs to pass into the UDF.
    Inputs are determined with the following priority: if the argument name in f

    1) is a key in argument_name_value_dict, the corresponding value of the
        dict is used (same value used for every row of the spark data frame)

    2) matches a column name in sdf, that column is used

    3) has a default specified in f, that value is used (same value used
        for every row of the spark dataframe)

    4) doesn't meet any of the above criteria, a ValueError is raised

    Third, the function returns the input spark dataframe with
    the added column that has values calculated by f with inputs
    determined by the priority above.

    If you get None results, make sure the data types you are using make sense.

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
    if new_col_name is None:
        new_col_name = f.__name__

    logger.debug(
        f"Adding column {new_col_name} to spark DataFrame using python function {f.__name__}"
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
    udf_inputs = []

    # list of sdf columns to confirm they exist if value not provided for arg
    sdf_columns = set(sdf.schema.names)

    # loop over args
    for idx, argument in enumerate(python_function_argument_names):
        if argument in constant_argument_name_value_dict:
            # if in argument_name_value_dict, use specified value
            udf_inputs.append(
                _constant_pyspark_col(
                    constant_argument_name_value_dict[argument]))
            logger.debug(
                f"Using dictionary supplied value {constant_argument_name_value_dict[argument]} for {argument}"
            )
        else:
            # otherwise if in sdf_columns, use the column's value
            if argument in sdf_columns:
                udf_inputs.append(argument)
                logger.debug(f"Using input spark dataframe column for {argument}")
            else:
                # otherwise if a default is specified in the function, use it
                if idx >= first_default_idx:
                    udf_inputs.append(
                        _constant_pyspark_col(
                            python_function_defaults[idx - first_default_idx]))
                    logger.debug(
                        f"Using function default value {python_function_defaults[idx - first_default_idx]} for {argument}"
                    )
                else:
                    # otherwise the user hasn't specified a required argument
                    raise ValueError(
                        f"{argument} not specified and not in sdf input")

    # convert python function to a UDF
    f_as_udf = pandas_udf(f, returnType=returnType, functionType=functionType)
    # add the column to the spark data frame using the arguments generated
    return sdf.withColumn(new_col_name, f_as_udf(*udf_inputs))