from unittest import TestCase

import pytest
from pyspark.sql.types import StringType


@pytest.fixture(scope='function', autouse=True)
def mock_udf_annotation(monkeypatch):
    def dummy_udf(f):
        return f

    def mock_udf(f=None, returnType=StringType()):
        return f if f else dummy_udf

    monkeypatch.setattr('pyspark.sql.functions.udf', mock_udf)


class TestUDFs(TestCase):

    def test_upper(self):
        """
        @udf(returnType=ArrayType(StringType()))
        def to_upper_list(s):
            return [i.upper() for i in s]
        """
        from our_package.spark import udfs as UDF
        self.assertEqual(UDF.to_upper_list(['potato', 'carrot', 'tomato']), ['POTATO', 'CARROT', 'TOMATO'])
