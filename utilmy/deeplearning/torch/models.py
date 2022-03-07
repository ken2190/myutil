
import os
import pandas as pd
import torch
import torch.nn as nn
import collections
from utilmy import log, log2
from abc import abstractmethod
from box import Box
import random
import numyp as np
class BaseModel(object):
    def __init__(self,arg):
        self.arg = Box(arg)
        self._device = self.device_setup(arg)
        self.net = self.create_model().to(self.device)
        

    def device_setup(self,arg):
        device = arg.device
        seed   = arg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if 'gpu' in device :
            try :
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception as e:
                log(e)
                device = 'cpu'
        return device

    def dataset(self,)-> pd.DataFrame:
        if os.path.isfile(self.arg.datapath):
            return pd.read_csv(self.arg.datapath)
        else:
            import wget
            wget.download(self.arg.dataurl)
            df = pd.read_csv(self.arg.datapath)
            return df
    
    @abstractmethod
    def create_model(self,):
      raise NotImplementedError
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def prepro_dataset(self,):
        raise NotImplementedError
    
    @abstractmethod
    def loss_calc(self,):
        raise NotImplementedError

    @abstractmethod
    def training(self,):
        raise NotImplementedError

    @property
    def device(self,):
        return self._device
    
    @device.setter
    def device(self,value):
        if isinstance(value,torch.device):
          self._device = value
        elif isinstance(value,str):
          self._device = torch.device(value)
        else:
          raise TypeError("device must be str or torch.device")

    def load_weight(self, path):
        # super().save_weight(path)()
        assert os.path.isfile(path),f"{path} does not exist"
        try:
          ckp = torch.load(path,map_location=self.device)
        except Exception as e:
          log(e)
          log(f"can't load the checkpoint from {path}")  
        if isinstance(ckp,collections.OrderedDict):
          self.net.load_state_dict(ckp)
        else:
          self.net.load_state_dict(ckp['state_dict'])
    
    def save_weight(self,path,meta_data=None):
      os.makedirs(os.path.dirname(path),exist_ok=True)
      ckp = {
          'state_dict':self.net.state_dict(),
      }
      if meta_data:
        if isinstance(meta_data,dict):
            ckp.update(meta_data)
        else:
            ckp.update({'meta_data':meta_data,})
            
        
      torch.save(path,ckp)

    def predict(self,x):
        # raise NotImplementedError
        output = self.net(x)
        return output 



class RuleEncoder_Create(BaseModel):
    def __init__(self, arg:dict):
        super(RuleEncoder_Create,self).__init__(arg)
        self.rule_ind = arg.rules.rule_ind

    def create_model(self):
        input_dim = self.arg.input_dim
        hidden_dim = self.arg.hidden_dim_encoder
        output_dim = self.arg.output_dim_encoder
        
        class RuleEncoder(torch.nn.Module):
            def __init__(self,dims=[16,4,1]):
                super(RuleEncoder, self).__init__()
                self.dims = dims 
                self.net = nn.Sequential()
                input_dim = dims[0]
                for layer_dim in dims[1:-1]:
                    self.net.append(nn.Linear(input_dim, layer_dim))
                    self.net.append(nn.ReLU())
                    input_dim = layer_dim
                self.net.append(nn.Linear(input_dim, dims[-1]))
                

            def forward(self, x):
                return self.net(x)

        return RuleEncoder(input_dim, output_dim, hidden_dim)   



class DataEncoder_Create(BaseModel):

    def __init__(self,arg):
        super(DataEncoder_Create,self).__init__(arg)

    def create_model(self):
        input_dim = self.arg.input_dim
        hidden_dim = self.arg.hidden_dim_encoder
        output_dim = self.arg.output_dim_encoder
        class DataEncoder(torch.nn.Module):
            def __init__(self,dims=[16,4,1]):
                super(DataEncoder, self).__init__()
                self.dims = dims 
                self.net = nn.Sequential()
                input_dim = dims[0]
                for layer_dim in dims[1:-1]:
                    self.net.append(nn.Linear(input_dim, layer_dim))
                    self.net.append(nn.ReLU())
                    input_dim = layer_dim
                self.net.append(nn.Linear(input_dim, dims[-1]))

            def forward(self, x):
                return self.net(x)

        return DataEncoder(input_dim,output_dim,hidden_dim)



class MergeEncoder_Create(BaseModel):


    """
    Dataset :  Raw Data  -->  dataloader

    DatasetModelRule  :  relatedt to rules (data + model part)

    DatasetModelTask  :  Base Model

    Modelmerge :   dataloader, Model1, Model2   -->  predict, train, ...
    """
    def __init__(self,arg):
        # self.rule_encoder = DatasetModelRule(arg).model
        # self.data_encoder = DatasetModelTask(arg).model
        super(MergeEncoder_Create,self).__init__(arg)
        
    def Dataset(rawdata:str) -> torch.utils.data.DataLoader:
        pass

    def create_model(self,):
        merge = self.arg.merge
        
        alpha = getattr(self.arg,'alpha',0)
        skip = getattr(self.arg,'skip',False)
        n_layers = self.arg.n_layers
        hidden_dim = self.arg.hidden_dim_db,
        output_dim = self.arg.output_dim
        self.rule_encoder = RuleEncoder_Create(self.arg)
        self.data_encoder = DataEncoder_Create(self.arg)
        dims = self.arg.MODEL_INFO.MODEL_MERGE.ARCHITECT.decoder
        class Modelmerge(torch.nn.Module):
            def __init__(self,rule_encoder,data_encoder,alpha,dims,merge,skip):
                super(Modelmerge, self).__init__()
                self.rule_encoder = rule_encoder.model
                self.data_encoder = data_encoder.model
                self.alpha = alpha
                self.merge = merge
                self.skip = skip
                if merge == 'cat':
                    self.input_dim_decision_block = self.rule_encoder.output_dim * 2
                elif merge == 'add':
                    self.input_dim_decision_block = self.rule_encoder.output_dim

                self.net = nn.Sequential()
                input_dim = dims[0]
                for layer_dim in dims[1:-1]:
                    self.net.append(nn.Linear(input_dim, layer_dim))
                    self.net.append(nn.ReLU())
                    input_dim = layer_dim
                self.net.append(nn.Linear(input_dim, dims[-1]))
                self.net.append(nn.Sigmoid())

                # self.net = nn.Sequential(*self.net)

            def forward(self, x):
            # merge: cat or add

                rule_z = self.rule_encoder(x)
                data_z = self.data_encoder(x)

                if self.merge == 'add':
                    z = self.alpha*rule_z + (1-alpha)*data_z
                elif self.merge == 'cat':
                    z = torch.cat((self.alpha*rule_z, (1-self.alpha)*data_z), dim=-1)
                elif self.merge == 'equal_cat':
                    z = torch.cat((rule_z, data_z), dim=-1)

                if self.skip:
                    if self.input_type == 'seq':
                        return self.net(z) + x[:, -1, :]
                    else:
                        return self.net(z) + x    # predict delta values
                else:
                    return self.net(z)    # predict absolute values
        return Modelmerge(self.rule_encoder,self.data_encoder,alpha,dims,merge,skip)
