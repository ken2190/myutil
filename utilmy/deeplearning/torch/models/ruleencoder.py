
from .base import BaseModel
import torch
import torch.nn as nn
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer,Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd



class RuleEncoder_Create(BaseModel):
    """

    Args:
        BaseModel (_type_): _description_
    """
    def __init__(self, arg:dict):
        super(RuleEncoder_Create,self).__init__(arg)
        self.rule_ind = arg.MODEL_INFO.MODEL_RULE.RULE_IND
        self.pert_coeff = arg.MODEL_INFO.MODEL_RULE.PERT
    def create_model(self):
        super(RuleEncoder_Create,self).create_model()
        dims = self.arg.MODEL_INFO.MODEL_RULE.ARCHITECT
        rule_ind = self.rule_ind
        pert_coeff = self.pert_coeff
        def get_perturbed_input(input_tensor, pert_coeff):
            '''
            X = X + pert_coeff*rand*X
            return input_tensor + input_tensor*pert_coeff*torch.rand()
            '''
            device = input_tensor.device
            result =  input_tensor + torch.abs(input_tensor)*pert_coeff*torch.rand(input_tensor.shape, device=device)
            return result

        class RuleEncoder(torch.nn.Module):
            def __init__(self,arg,dims=[20,100,16]):
                super(RuleEncoder, self).__init__()
                self.dims = dims 
                self.output_dim = dims[-1]
                # self.net = nn.Sequential()
                self.net = []
                input_dim = dims[0]
                for layer_dim in dims[1:-1]:

                    self.net.append(torch.nn.Linear(input_dim, layer_dim))

                    self.net.append(torch.nn.ReLU())
                    input_dim = layer_dim
                self.net.append(torch.nn.Linear(input_dim, dims[-1]))
                self.net = torch.nn.Sequential(*self.net)

            def forward(self, x,**kwargs):

                x[:,rule_ind] = get_perturbed_input(x[:,rule_ind], pert_coeff)

                return self.net(x)

        return RuleEncoder(self.arg,dims)   

    def create_loss(self,) -> torch.nn.Module:
        super(RuleEncoder_Create,self).create_loss()
        class LossRule(torch.nn.Module):
            
            def __init__(self):
                super(LossRule,self).__init__()
                self.relu = torch.nn.ReLU()

            def forward(self,output,target):
                return torch.mean(self.relu(output-target))

        return LossRule()

    def load_DataFrame(self,) -> pd.DataFrame:
        from sklearn.datasets import fetch_covtype
        df = fetch_covtype(return_X_y=False, as_frame=True)
        df =df.data
        #   log(df)
        #   log(df.columns)
        df = df.iloc[:500, :10]
        #   log(df)
        return df

    def prepro_dataset(self,df):
        if df is None:
            df = self.df
        coly  = 'Slope'  # df.columns[-1]
        y_raw = df[coly]
        X_raw = df.drop([coly], axis=1)

        X_column_trans = ColumnTransformer(
                [(col, StandardScaler() if not col.startswith('Soil_Type') else Binarizer(), [col]) for col in X_raw.columns],
                remainder='passthrough')

        y_trans = StandardScaler()

        X = X_column_trans.fit_transform(X_raw)
        # y = y_trans.fit_transform(y_raw.array.reshape(1, -1))
        y = y_trans.fit_transform(y_raw.values.reshape(-1, 1))

        ### Binarize
        y = np.array([  1 if yi >0.5 else 0 for yi in y])

        seed= 42
        train_X, test_X, train_y, test_y = train_test_split(X,  y,  test_size=1 - self.arg.TRAINING_CONFIG.TRAIN_RATIO, random_state=seed)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= self.arg.TRAINING_CONFIG.TEST_RATIO / (self.arg.TRAINING_CONFIG.TEST_RATIO + self.arg.TRAINING_CONFIG.VAL_RATIO), random_state=seed)
        # print(np.float32(train_X).shape)
        # exit()
        return (np.float32(train_X), np.float32(train_y), np.float32(valid_X), np.float32(valid_y), np.float32(test_X), np.float32(test_y) )