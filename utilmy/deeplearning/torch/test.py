from turtle import forward
import models
from config import ARG
import numpy as np
import torch
from models.utils import *
import copy 
def test():
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

def test2():
    ARG_copy = copy.copy(ARG)
    ARG_copy.MODEL_INFO.MODEL_RULE.ARCHITECT = [9,100,16]
    ARG_copy.MODEL_INFO.MODEL_TASK.ARCHITECT = [9,100,16]
    ARG_copy.TRAINING_CONFIG.SAVE_FILENAME = './model_x9.pt'
    load_DataFrame = models.RuleEncoder_Create.load_DataFrame   
    prepro_dataset = models.RuleEncoder_Create.prepro_dataset
    model = models.MergeEncoder_Create(ARG_copy)
    model.build()        
    model.training(load_DataFrame,prepro_dataset) 

def inference():
    model = models.MergeEncoder_Create(ARG)
    model.build()
    model.load_weights('model.pt')
    inputs = torch.randn((1,20)).to(model.device)
    outputs = model.predict(inputs)
    print(outputs)
if __name__ =='__main__':

    test()
    test2()
    inference()