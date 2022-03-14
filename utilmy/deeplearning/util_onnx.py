# -*- coding: utf-8 -*-
MNAME = "utilmy.deeplearning.util_onnx"
HELP = """ utils for ONNX runtime Optimization


cd myutil
python utilmy/deeplearning/util_onnx.py    test1




"""
import os, numpy as np, glob, pandas as pd, matplotlib.pyplot as plt
from box import Box


#### Types
from numpy import ndarray
from typing import List, Optional, Tuple, Union


#############################################################################################
from utilmy import log, log2


def help():
    """function help        
    """
    from utilmy import help_create
    print( HELP + help_create(MNAME) )



#############################################################################################
def test_all() -> None:
    """function test_all   to be used in test.py        
    """
    log(MNAME)
    test1()


def test1() -> None:
    """function test1     
    """
    d = Box({})



########################################################################################################
############## Core Code ###############################################################################
def onnx_convert():
  pass




def onnx_load_model():
  pass



def onnx_load_onnx():
  pass



def onnx_validate_onnx():
   pass


















#############################################################################################
#############################################################################################

if 'utils':
    def load_function_uri(uri_name="path_norm"):
        """ Load dynamically function from URI
        ###### Pandas CSV case : Custom MLMODELS One
        #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"
        ###### External File processor :
        #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
        """
        import importlib, sys
        from pathlib import Path
        pkg = uri_name.split(":")

        assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
        package, name = pkg[0], pkg[1]
        
        try:
            #### Import from package mlmodels sub-folder
            return  getattr(importlib.import_module(package), name)

        except Exception as e1:
            try:
                ### Add Folder to Path and Load absoluate path module
                path_parent = str(Path(package).parent.parent.absolute())
                sys.path.append(path_parent)
                #log(path_parent)

                #### import Absolute Path model_tf.1_lstm
                model_name   = Path(package).stem  # remove .py
                package_name = str(Path(package).parts[-2]) + "." + str(model_name)
                #log(package_name, model_name)
                return  getattr(importlib.import_module(package_name), name)

            except Exception as e2:
                raise NameError(f"Module {pkg} notfound, {e1}, {e2}")






###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()


