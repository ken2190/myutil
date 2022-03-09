from turtle import forward
import models
from config import ARG
import unittest
import torch
from models.utils import *
class Test_Model(unittest.TestCase):
    # model.RuleEncoder_Create.prepro_dataset = ''  
    def test_init(self):  
        model = models.RuleEncoder_Create(ARG)
        model.build()
        self.assertIsInstance(model.net,torch.nn.Module)
        model = models.DataEncoder_Create(ARG)
        model.build()
        self.assertIsInstance(model.net,torch.nn.Module)
        model = models.MergeEncoder_Create(ARG)
        model.build()
        self.assertIsInstance(model.net,torch.nn.Module)

    def test_dims(self,):
        pass

class Test_Function(unittest.TestCase):
    
    def test_load_data(self):
        model = models.MergeEncoder_Create(ARG)
        model.build()        
        model.training()
    
def test2():
    """
    load and process data from default dataset
    if you want to training with custom datase.
    Do following step:
    def load_DataFrame(path) -> pandas.DataFrame:
        ...
        ...
        return df
    def prepro_dataset(df) -> tuple:
        ...
        ...
        return TrainX,trainY,...
    
    """
    load_DataFrame = None
    prepro_dataset = None
    model = models.MergeEncoder_Create(ARG)
    model.build()        
    model.training(load_DataFrame,prepro_dataset)   



if __name__ =='__main__':
    unittest.main()    
    test2()