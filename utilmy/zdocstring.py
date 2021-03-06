# -*- coding: utf-8 -*-
"""Example Google style docstrings.
    Example:

        Actual Docstring code ::

          This is an example of a module level function.

          Args:
              param1 (int): The first parameter.
              param2 (:obj:`str`, optional): The second parameter. Defaults to None.
                  Second line of description should be indented.
              *args: Variable length argument list.
              **kwargs: Arbitrary keyword arguments.

          Returns:
              bool: True if successful, False otherwise.


          Example:
              including literal blocks::

                 ok = np.cos(5)


"""



import os, sys, numpy as np



#####################################################################
def a__google_doc_string_example(package:str="mlmodels.util", name:str="path_norm"):
    """This is an example of a module level function.
    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.
        Following lines should be indented to match the first line.
        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {   'param1': param1,
                'param2': param2
            }

    Example:
        Usage 1::

          import numpy as np
          a  = np.cos(1)

        Usage 2::

          This is an example of a module level function.
          Args:
              param1 (int): The first parameter.
              param2 (:obj:`str`, optional): The second parameter. Defaults to None.
                  Second line of description should be indented.
              *args: Variable length argument list.
              **kwargs: Arbitrary keyword arguments.

          Returns:
              bool: True if successful, False otherwise.


          Example:
              including literal blocks::

                 ok = np.cos(5)



    """
    import importlib
    return  getattr(importlib.import_module(package), name)



class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.
    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).
    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.
    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.
    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.
        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.
        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.
        Note:
            Do not include the `self` parameter in the ``Args`` section.
        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.
        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ['attr4']

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return 'readonly_property'

    @property
    def readwrite_property(self):
        """:obj:`list` of :obj:`str`: Properties with both a getter and setter
        should only be documented in their getter method.
        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.
        Note:
            Do not include the `self` parameter in the ``Args`` section.
        Args:
            param1: The first parameter.
            param2: The second parameter.
        Returns:
            True if successful, False otherwise.
        """
        return True

    def __special__(self):
        """By default special members with docstrings are not included.
        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.
        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::
            napoleon_include_special_with_doc = True
        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.
        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.
        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::
            napoleon_include_private_with_doc = True
        """
        pass

    def _private_without_docstring(self):
        pass