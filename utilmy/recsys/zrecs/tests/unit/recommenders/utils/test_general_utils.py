# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from recommenders.utils.general_utils import invert_dictionary, get_number_processors


def test_invert_dictionary():
    """function test_invert_dictionary.
    Doc::
            
            Args:
            Returns:
                
    """
    d = {"a": 1, "b": 2}
    d_inv = invert_dictionary(d)
    assert d_inv == {1: "a", 2: "b"}


def test_get_number_processors():
    """function test_get_number_processors.
    Doc::
            
            Args:
            Returns:
                
    """
    assert get_number_processors() >= 1
