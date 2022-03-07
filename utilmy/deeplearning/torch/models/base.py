
import os
import pandas as pd
import torch
import torch.nn as nn
import collections
from utilmy import log, log2
from abc import abstractmethod
from box import Box
import random
import numpy as np
class BaseModel(object):
    def __init__(self,arg):
        self.arg = Box(arg)
        self._device = self.device_setup(arg)
        # self.net = self.create_model().to(self.device)
        self.losser = None
        self.is_train = False
        
    @abstractmethod
    def create_model(self,) -> torch.nn.Module:
      raise NotImplementedError
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def prepro_dataset(self,csv) -> pd.DataFrame:
        raise NotImplementedError
    
    @abstractmethod
    def create_loss(self,) -> torch.nn.Module:
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

    def build(self,):
        self.net = self.create_model().to(self.device)
        self.criterior = self.create_loss.to(self.device)
        self.is_train = False
    
    def train(self): #equavilant model.train() in pytorch
        self.is_train = True
        self.net.train()

    def eval(self):
        self.is_train = False
        self.net.eval()

    def device_setup(self,arg):
        device = getattr(arg,'device','cpu')
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

    def load_DataFrame(self,path=None)-> pd.DataFrame:
        if path:
            self.df = pd.read_csv(path)
            return self.df
        if os.path.isfile(self.arg.datapath):
            return pd.read_csv(self.arg.datapath)
        else:
            import requests
            import io
            r = requests.get(self.arg.dataurl)
            if r.status_code =='200':
                df = pd.read_csv(io.BytesIO(r.content))
            else:
                raise Exception("Can't read data")
            self.df = df
            return df


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

    def predict(self,x,**kwargs):
        # raise NotImplementedError
        output = self.net(x,**kwargs)
        return output 


