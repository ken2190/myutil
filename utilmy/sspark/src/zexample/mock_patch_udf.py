from unittest import TestCase

def dummy_udf(f):
    return f

def mock_udf(f=None, returnType=None):
    return f if f else dummy_udf

from mock import patch
patch('pyspark.sql.functions.udf', mock_udf).start()

from our_package.spark import udfs as UDF


class TestUDFs(TestCase):

    def test_upper(self):
        """
        @udf(returnType=ArrayType(StringType()))
        def to_upper_list(s):
            return [i.upper() for i in s]
        """
        self.assertEqual(UDF.to_upper_list(['potato', 'carrot', 'tomato']), ['POTATO', 'CARROT', 'TOMATO'])
