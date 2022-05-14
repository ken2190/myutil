
    def _test_spark_udtf(self):
        """
        # source
        root
         |-- id: long (nullable = true)
         |-- title: string (nullable = true)
         |-- abstract: string (nullable = true)
         |-- content: string (nullable = true)
         |-- else: string (nullable = true)
        
        +---+-----+--------+-------+----+
        | id|title|abstract|content|else|
        +---+-----+--------+-------+----+
        |  0|   t1|      c1|   xxx1| ...|
        |  1|   t2|      c2|   xxx2| ...|
        |  2|   t3|      c3|   xxx3| ...|
        +---+-----+--------+-------+----+
        
        # middle udf result
        +---+-----+--------+-------+----+--------------------+
        | id|title|abstract|content|else|              table1|
        +---+-----+--------+-------+----+--------------------+
        |  0|   t1|      c1|   xxx1| ...|[[0, t1], [0, c1]...|
        |  1|   t2|      c2|   xxx2| ...|[[1, t2], [1, c2]...|
        |  2|   t3|      c3|   xxx3| ...|[[2, t3], [2, c3]...|
        +---+-----+--------+-------+----+--------------------+
        
        
        # result
        +---+----+
        | id| col|
        +---+----+
        |  0|  t1|
        |  0|  c1|
        |  0|xxx1|
        |  1|  t2|
        |  1|  c2|
        |  1|xxx2|
        |  2|  t3|
        |  2|  c3|
        |  2|xxx3|
        +---+----+
        
        root
         |-- id: string (nullable = true)
         |-- col: string (nullable = true)
        

        :return: 
        """
        cols = ['id', 'title', 'abstract', 'content', 'else']
        data = [
            (0, 't1', 'c1', 'xxx1', '...'),
            (1, 't2', 'c2', 'xxx2', '...'),
            (2, 't3', 'c3', 'xxx3', '...')
        ]
        df = self.spark.createDataFrame(data, cols)
        df.printSchema()
        df.show()

        import pyspark.sql.types as T
        import pyspark.sql.functions as F

        cols = ["id", "col"]
        fields = []
        for col in cols:
            fields.append(T.StructField(col, T.StringType(), True))

        @F.udf(returnType=T.ArrayType(T.StructType(fields)))
        def tb1(id, title, abstract, content):
            return [
                {"id": str(id), "col": title},
                {"id": str(id), "col": abstract},
                {"id": str(id), "col": content}
            ]
        df1 = df.withColumn("table1", tb1(F.col('id'), F.col('title'), F.col('abstract'), F.col('content')))
        df1.show()
        df1 = df1.select(F.explode(F.col("table1")).alias("expload_col"))
        for col in cols:
            df1 = df1.withColumn(col, F.col('expload_col.%s' % col))
        df1 = df1.select(cols)
        df1.show()
        df1.printSchema()