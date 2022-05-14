from importlib import reload
from unittest import TestCase

from mock import patch
from our_package.spark import udfs as UDF


def dummy_udf(f):
    return f


def mock_udf(f=None, returnType=None):
    return f if f else dummy_udf


class TestUDFs(TestCase):
    udf_patch = patch('pyspark.sql.functions.udf', mock_udf)

    @classmethod
    def setUpClass(cls):
        cls.udf_patch.start()
        reload(UDF)

    @classmethod
    def tearDownClass(cls):
        cls.udf_patch.stop()
        reload(UDF)
    
    def test_upper(self):
        """
        @udf(returnType=ArrayType(StringType()))
        def to_upper_list(s):
            return [i.upper() for i in s]
        """
        self.assertEqual(UDF.to_upper_list(['potato', 'carrot', 'tomato']), ['POTATO', 'CARROT', 'TOMATO'])
     