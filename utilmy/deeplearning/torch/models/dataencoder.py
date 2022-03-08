
from .base import BaseModel
import torch
import torch.nn as nn
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer,Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
class DataEncoder_Create(BaseModel):
    """
    DataEncoder

    Method:
        create_model : 
        create_loss : 
    """
    def __init__(self,arg):
        super(DataEncoder_Create,self).__init__(arg)

    def create_model(self):
        dims = self.arg.MODEL_INFO.MODEL_TASK.ARCHITECT
        class DataEncoder(torch.nn.Module):
            def __init__(self,dims=[20,100,16]):
                super(DataEncoder, self).__init__()
                self.dims = dims 
                self.output_dim = dims[-1]
                # self.net = nn.Sequential()
                self.net = []
                input_dim = dims[0]
                for layer_dim in dims[:-1]:
                    self.net.append(nn.Linear(input_dim, layer_dim))
                    self.net.append(nn.ReLU())
                    input_dim = layer_dim
                self.net.append(nn.Linear(input_dim, dims[-1]))
                self.net = nn.Sequential(*self.net)

            def forward(self, x,**kwargs):
                return self.net(x)

        return DataEncoder(dims)

    def create_loss(self) -> torch.nn.Module:
        return torch.nn.BCELoss()