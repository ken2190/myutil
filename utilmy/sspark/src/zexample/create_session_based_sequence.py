from pyspark.sql.functions import collect_list


def create_session_based_sequence(
   dataframe,
   spark_session,
   session_id_col,
   seq_col,
   page_name,
   output_col,
   ) -> DataFrame:
   # aggregate the dataframe to create a list for every session, containing (page sequence, page id) pairs for that
   # session
   session_dataframe = (
       dataframe.select([session_id_col] + [seq_col, page_name])
       .groupBy(session_id_col)
       .agg(collect_list(struct(seq_col, page_name)).alias("seq_and_page_name_tuple"))
   )

   # sort the list by page sequence column and output the page_id
   sort_by_first_udf = create_sort_by_first_udf(spark_session=spark_session)
   return session_dataframe.select(
       col(session_id_col),
       sort_by_first_udf("seq_and_page_name_tuple").alias(output_col),
   )

def create_sort_by_first_udf(spark_session) -> udf:
   """
   This method create a user-define function that sorts a tuples according to the first element of the tuple,
   return a list containing the second element.  For example, if the list of tuples is:
       [(3, 'a'), (2, 'b'), (1, 'c')]
   The udf would return
       ['c', 'b', 'a']
   This is because the order of the first index is 1, 2, 3

   :param spark_session: the current spark session
   :type spark_session: SparkSession
   :return: the sort_by_first udf
   :rtype: udf
   """

   def sort_by_first(a_tuple):
       sorted_tuple = sorted(a_tuple, key=operator.itemgetter(0))
       return [item[1] for item in sorted_tuple]

   sort_by_first_udf = udf(
       f=sort_by_first, returnType=ArrayType(elementType=StringType())
   )
   spark_session.udf.register("sort_by_first_udf", sort_by_first_udf)
   return sort_by_first_udf