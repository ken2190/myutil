# -*- coding: utf-8 -*-
MNAME = "utilmy.deeplearning.torch.rule_encoder"
HELP = """ utils for model explanation

### pip install fire

python rule_encoder3.py test1


"""
import os,sys, collections, random, numpy as np,  glob, pandas as pd, matplotlib.pyplot as plt ;from box import Box
from copy import deepcopy
from abc import abstractmethod

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


from torch.utils.data import DataLoader, TensorDataset

from turtle import forward
from sklearn.utils import shuffle
from tqdm import tqdm



#### Types


#############################################################################################
from utilmy import log, log2

def help():
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    log(ss)


#############################################################################################
def test_all():
    log(MNAME)
    test()
    # test2()



def tes1():    
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
    from box import Box ; from copy import deepcopy
    BATCH_SIZE = None
    ARG = Box({
        'DATASET': {},
        'MODEL_INFO' : {},
        'TRAINING_CONFIG' : {},
    })


    MODEL_ZOO = {
        'dataonly': {'rule': 0.0},
        'ours-beta1.0': {'beta': [1.0], 'scale': 1.0, 'lr': 0.001},
        'ours-beta0.1': {'beta': [0.1], 'scale': 1.0, 'lr': 0.001},
        'ours-beta0.1-scale0.1': {'beta': [0.1], 'scale': 0.1},
        'ours-beta0.1-scale0.01': {'beta': [0.1], 'scale': 0.01},
        'ours-beta0.1-scale0.05': {'beta': [0.1], 'scale': 0.05},
        'ours-beta0.1-pert0.001': {'beta': [0.1], 'pert': 0.001},
        'ours-beta0.1-pert0.01': {'beta': [0.1], 'pert': 0.01},
        'ours-beta0.1-pert0.1': {'beta': [0.1], 'pert': 0.1},
        'ours-beta0.1-pert1.0': {'beta': [0.1], 'pert': 1.0},
                    
    }


    ### ARG.DATASET
    ARG.seed = 42
    ARG.DATASET.PATH =  './cardio_train.csv'
    ARG.DATASET.URL = 'https://github.com/caravanuden/cardio/raw/master/cardio_train.csv'



    #ARG.TRAINING_CONFIG
    ARG.TRAINING_CONFIG = Box()
    ARG.TRAINING_CONFIG.SEED = 42
    ARG.TRAINING_CONFIG.DEVICE = 'cpu'
    ARG.TRAINING_CONFIG.BATCH_SIZE = 32
    ARG.TRAINING_CONFIG.EPOCHS = 1
    ARG.TRAINING_CONFIG.EARLY_STOPPING_THLD = 10
    ARG.TRAINING_CONFIG.VALID_FREQ = 1
    ARG.TRAINING_CONFIG.SAVE_FILENAME = './model.pt'
    ARG.TRAINING_CONFIG.TRAIN_RATIO = 0.7
    ARG.TRAINING_CONFIG.VAL_RATIO = 0.2
    ARG.TRAINING_CONFIG.TEST_RATIO = 0.1

    ### ARG.MODEL_INFO

    ARG.MODEL_INFO.TYPE = 'dataonly' 
    PARAMS = MODEL_ZOO[ARG.MODEL_INFO.TYPE]
    ARG.MODEL_INFO.LR = PARAMS.get('lr',None)




    ### ARG.MODEL_RULE
    ARG.MODEL_INFO.MODEL_RULE = Box()   #MODEL_RULE
    ARG.MODEL_INFO.MODEL_RULE.RULE = PARAMS.get('rule',None)
    ARG.MODEL_INFO.MODEL_RULE.SCALE = PARAMS.get('scale',1.0)
    ARG.MODEL_INFO.MODEL_RULE.PERT = PARAMS.get('pert',0.1)
    ARG.MODEL_INFO.MODEL_RULE.BETA = PARAMS.get('beta',[1.0])
    beta_param = ARG.MODEL_INFO.MODEL_RULE.BETA

    from torch.distributions.beta import Beta
    if   len(beta_param) == 1:  ARG.MODEL_INFO.MODEL_RULE.ALPHA_DIST = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:  ARG.MODEL_INFO.MODEL_RULE.ALPHA_DIST = Beta(float(beta_param[0]), float(beta_param[1]))

    ARG.MODEL_INFO.MODEL_RULE.NAME = ''
    ARG.MODEL_INFO.MODEL_RULE.RULE_IND = 2
    ARG.MODEL_INFO.MODEL_RULE.RULE_THRESHOLD = 129.5
    ARG.MODEL_INFO.MODEL_RULE.SRC_OK_RATIO = 0.3
    ARG.MODEL_INFO.MODEL_RULE.SRC_UNOK_RATIO = 0.7
    ARG.MODEL_INFO.MODEL_RULE.TARGET_RULE_RATIO = 0.7
    ARG.MODEL_INFO.MODEL_RULE.ARCHITECT = [
        20,
        100,
        16
    ]


    ### ARG.MODEL_TASK
    ARG.MODEL_INFO.MODEL_TASK = Box()   #MODEL_TASK
    ARG.MODEL_INFO.MODEL_TASK.NAME = ''
    ARG.MODEL_INFO.MODEL_TASK.ARCHITECT = [
        20,
        100,
        16
    ]

    ARG.MODEL_INFO.MODEL_MERGE = Box()
    ARG.MODEL_INFO.MODEL_MERGE.NAME = ''
    ARG.MODEL_INFO.MODEL_MERGE.SKIP = False
    ARG.MODEL_INFO.MODEL_MERGE.MERGE = 'cat'
    ARG.MODEL_INFO.MODEL_MERGE.ARCHITECT = {
        'decoder': [
        32,  
        100,
        1
    ]
    }

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
    ARG_copy = deepcopy(ARG)
    ARG_copy.MODEL_INFO.MODEL_RULE.ARCHITECT = [9,100,16]
    ARG_copy.MODEL_INFO.MODEL_TASK.ARCHITECT = [9,100,16]
    ARG_copy.TRAINING_CONFIG.SAVE_FILENAME = './model_x9.pt'
    load_DataFrame = models.RuleEncoder_Create.load_DataFrame   
    prepro_dataset = models.RuleEncoder_Create.prepro_dataset
    model = models.MergeEncoder_Create(ARG_copy)
    model.build()        
    model.training(load_DataFrame,prepro_dataset) 

    model.save_weight('ztmp/model_x9/') 
    model.load_weights('ztmp/model_x9.pt')
    inputs = torch.randn((1,20)).to(model.device)
    outputs = model.predict(inputs)
    print(outputs)




##############################################################################################
def dataloader_create(train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,device=None,batch_size=None):
    # batch_size = batch_size
    train_loader, valid_loader, test_loader = None, None, None

    if train_X is not None :
        train_X, train_y = torch.tensor(train_X, dtype=torch.float32, device=device), torch.tensor(train_y, dtype=torch.float32, device=device)
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        # log("data size", len(train_X) )

    if valid_X is not None :
        valid_X, valid_y = torch.tensor(valid_X, dtype=torch.float32, device=device), torch.tensor(valid_y, dtype=torch.float32, device=device)
        valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
        # log("data size", len(valid_X)  )

    if test_X  is not None :
        test_X, test_y   = torch.tensor(test_X,  dtype=torch.float32, device=device), torch.tensor(test_y, dtype=torch.float32, device=device)
        test_loader  = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])
        # log("data size:", len(test_X) )

    return train_loader, valid_loader, test_loader



##############################################################################################
class MergeEncoder_Create(BaseModel):
    """
    """
    def __init__(self,arg):
        super(MergeEncoder_Create,self).__init__(arg)
        self.rule_encoder = RuleEncoder_Create(self.arg)
        self.data_encoder = DataEncoder_Create(self.arg)

    def create_model(self,):
        # merge = self.arg.merge
        merge = getattr(self.arg.MODEL_INFO.MODEL_MERGE,'MERGE','add')
        skip = getattr(self.arg.MODEL_INFO.MODEL_MERGE,'SKIP',False)
        
        dims = self.arg.MODEL_INFO.MODEL_MERGE.ARCHITECT.decoder
        class Modelmerge(torch.nn.Module):
            def __init__(self,rule_encoder,data_encoder,dims,merge,skip):
                super(Modelmerge, self).__init__()
                self.rule_encoder = rule_encoder.net
                self.data_encoder = data_encoder.net
                self.merge = merge
                self.skip = skip
                if merge == 'cat':
                    self.input_dim_decision_block = self.rule_encoder.output_dim * 2
                elif merge == 'add':
                    self.input_dim_decision_block = self.rule_encoder.output_dim

                # self.net = nn.Sequential()
                self.net = []
                input_dim = dims[0]
                for layer_dim in dims[1:-1]:
                    self.net.append(nn.Linear(input_dim, layer_dim))
                    self.net.append(nn.ReLU())
                    input_dim = layer_dim
                self.net.append(nn.Linear(input_dim, dims[-1]))
                self.net.append(nn.Sigmoid())

                self.net = nn.Sequential(*self.net)

            def forward(self, x,**kwargs):
            # merge: cat or add
                alpha = kwargs.get('alpha',0) # default only use rule_z
                # scale = kwargs.get('scale',1)
                rule_z = self.rule_encoder(x)
                
                data_z = self.data_encoder(x)

                if self.merge == 'add':
                    z = alpha*rule_z + (1-alpha)*data_z
                elif self.merge == 'cat':
                    z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)
                elif self.merge == 'equal_cat':
                    z = torch.cat((rule_z, data_z), dim=-1)

                if self.skip:
                    if self.input_type == 'seq':
                        return self.net(z) + x[:, -1, :]
                    else:
                        return self.net(z) + x    # predict delta values
                else:
                    return self.net(z)    # predict absolute values
        return Modelmerge(self.rule_encoder,self.data_encoder,dims,merge,skip)

    def build(self):
        # super(MergeEncoder_Create,self).build()
        self.rule_encoder.build()
        self.data_encoder.build()
        self.criterior = self.create_loss().to(self.device)
        self.net = self.create_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        
    def create_loss(self,):
        rule_criterior = self.rule_encoder.criterior 
        data_criterior = self.data_encoder.criterior
        class MergeLoss(torch.nn.Module):

            def __init__(self,rule_criterior,data_criterior):
                super(MergeLoss,self).__init__()
                self.rule_criterior = rule_criterior
                self.data_criterior = data_criterior

            def forward(self,output,target,alpha=0,scale=1):
                
                rule_loss = self.rule_criterior(output,target)
                data_loss = self.data_criterior(output,target.reshape(output.shape))                
                result = rule_loss * alpha * scale + data_loss * (1-alpha)
                return result

        return MergeLoss(rule_criterior,data_criterior)

    def prepro_dataset(self,df=None):
        if not df:              
            df = self.df     # if there is no dataframe feeded , get df from model itself

        coly = 'cardio'
        y     = df[coly]
        X_raw = df.drop([coly], axis=1)

        # log("Target class ratio:")
        # log("# of y=1: {}/{} ({:.2f}%)".format(np.sum(y==1), len(y), 100*np.sum(y==1)/len(y)))
        # log("# of y=0: {}/{} ({:.2f}%)\n".format(np.sum(y==0), len(y), 100*np.sum(y==0)/len(y)))

        column_trans = ColumnTransformer(
            [('age_norm', StandardScaler(), ['age']),
            ('height_norm', StandardScaler(), ['height']),
            ('weight_norm', StandardScaler(), ['weight']),
            ('gender_cat', OneHotEncoder(), ['gender']),
            ('ap_hi_norm', StandardScaler(), ['ap_hi']),
            ('ap_lo_norm', StandardScaler(), ['ap_lo']),
            ('cholesterol_cat', OneHotEncoder(), ['cholesterol']),
            ('gluc_cat', OneHotEncoder(), ['gluc']),
            ('smoke_cat', OneHotEncoder(), ['smoke']),
            ('alco_cat', OneHotEncoder(), ['alco']),
            ('active_cat', OneHotEncoder(), ['active']),
            ], remainder='passthrough'
        )

        X = column_trans.fit_transform(X_raw)
        nsamples = X.shape[0]
        X_np = X.copy()


        ######## Rule : higher ap -> higher risk   #####################################
        """  Identify Class y=0 /1 from rule 1

        """
        if 'rule1':
            rule_threshold = self.arg.MODEL_INFO.MODEL_RULE.RULE_THRESHOLD
            rule_ind       = self.arg.MODEL_INFO.MODEL_RULE.RULE_IND
            rule_feature   = 'ap_hi'
            src_unok_ratio = self.arg.MODEL_INFO.MODEL_RULE.SRC_OK_RATIO
            src_ok_ratio   = self.arg.MODEL_INFO.MODEL_RULE.SRC_UNOK_RATIO

            #### Ok cases: nornal
            low_ap_negative  = (df[rule_feature] <= rule_threshold) & (df[coly] == 0)    # ok
            high_ap_positive = (df[rule_feature] > rule_threshold)  & (df[coly] == 1)    # ok

            ### Outlier cases (from rule)
            low_ap_positive  = (df[rule_feature] <= rule_threshold) & (df[coly] == 1)    # unok
            high_ap_negative = (df[rule_feature] > rule_threshold)  & (df[coly] == 0)    # unok




        #### Merge rules ##############################################
        # Samples in ok group
        idx_ok = low_ap_negative | high_ap_positive


        # Samples in Unok group
        idx_unok = low_ap_negative | high_ap_positive



        ##############################################################################
        # Samples in ok group
        X_ok = X[ idx_ok ]
        y_ok = y[ idx_ok ]
        y_ok = y_ok.to_numpy()
        X_ok, y_ok = shuffle(X_ok, y_ok, random_state=0)
        num_ok_samples = X_ok.shape[0]


        # Samples in Unok group
        X_unok = X[ idx_unok ]
        y_unok = y[ idx_unok ]
        y_unok = y_unok.to_numpy()
        X_unok, y_unok = shuffle(X_unok, y_unok, random_state=0)
        num_unok_samples = X_unok.shape[0]


        ######### Build a source dataset
        n_from_unok = int(src_unok_ratio * num_unok_samples)
        n_from_ok   = int(n_from_unok * src_ok_ratio / (1- src_ok_ratio))

        X_src = np.concatenate((X_ok[:n_from_ok], X_unok[:n_from_unok]), axis=0)
        y_src = np.concatenate((y_ok[:n_from_ok], y_unok[:n_from_unok]), axis=0)

        # log("Source dataset statistics:")
        # log("# of samples in ok group: {}".format(n_from_ok))
        # log("# of samples in Unok group: {}".format(n_from_unok))
        # log("ok ratio: {:.2f}%".format(100 * n_from_ok / (X_src.shape[0])))


        ##### Split   #########################################################################
        seed= 42
        train_X, test_X, train_y, test_y = train_test_split(X_src,  y_src,  test_size=1 - self.arg.TRAINING_CONFIG.TRAIN_RATIO, random_state=seed)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= self.arg.TRAINING_CONFIG.TEST_RATIO / (self.arg.TRAINING_CONFIG.TEST_RATIO + self.arg.TRAINING_CONFIG.VAL_RATIO), random_state=seed)
        return (train_X, train_y, valid_X,  valid_y, test_X,  test_y, )
        

    def training(self,):
        self.train()
        self.load_DataFrame() #load from config
        
        train_X, train_y, valid_X,  valid_y, test_X,  test_y, = self.prepro_dataset()
        train_loader, valid_loader, test_loader =  dataloader_create(train_X, train_y, 
                                                                    valid_X,  valid_y,
                                                                    test_X,  test_y,
                                                                    device=self.device,
                                                                    batch_size=self.arg.TRAINING_CONFIG.BATCH_SIZE)

        EPOCHS = self.arg.TRAINING_CONFIG.EPOCHS
        
        n_train = len(train_loader)
        n_val = len(valid_loader)
        
        for epoch in tqdm(range(1,EPOCHS+1)):

            loss_train = 0
            for inputs,targets in tqdm(train_loader,total=n_train, desc='training'):
                if   self.arg.MODEL_INFO.TYPE.startswith('dataonly'):  alpha = 0.0
                elif self.arg.MODEL_INFO.TYPE.startswith('ruleonly'):  alpha = 1.0
                elif self.arg.MODEL_INFO.TYPE.startswith('ours'):      alpha = self.arg.MODEL_INFO.MODEL_RULE.ALPHA_DIST.sample().item()
                predict = self.predict(inputs,alpha=alpha)
                self.optimizer.zero_grad()
                scale =1
                loss = self.criterior(predict,targets,alpha=alpha,scale=scale)
                loss.backward()
                self.optimizer.step()
                loss_train += loss * inputs.size(0)
            loss_train /= len(train_loader.dataset) # mean on dataset

            loss_val = 0
            with torch.no_grad():
                for inputs,targets in tqdm(valid_loader, total=n_val, desc='validating'):
                    predict = self.predict(inputs)
                    self.optimizer.zero_grad()
                    scale=1
                    loss = self.criterior(predict,targets,alpha=alpha,scale=scale)
                    
                    loss_val += loss * inputs.size(0)
            loss_val /= len(valid_loader.dataset) # mean on dataset

            self.save_weight(
                path = self.arg.TRAINING_CONFIG.SAVE_FILENAME,
                meta_data = {
                    'epoch' : epoch,
                    'loss_train': loss_train,
                    'loss_val': loss_val,
                }
            )





##############################################################################################
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
        if not df:
            df = self.df
        coly  = 'Slope'  # df.columns[-1]
        y_raw = df[coly]
        X_raw = df.drop([coly], axis=1)

        X_column_trans = ColumnTransformer(
                [(col, StandardScaler() if not col.startswith('Soil_Type') else Binarizer(), [col]) for col in X_raw.columns],
                remainder='passthrough')

        y_trans = StandardScaler()

        X = X_column_trans.fit_transform(X_raw)
        y = y_trans.fit_transform(y_raw.array.reshape(-1, 1))

        ### Binarize
        y = np.array([  1 if yi >0.5 else 0 for yi in y])

        seed= 42
        train_X, test_X, train_y, test_y = train_test_split(X,  y,  test_size=1 - self.arg.train_ratio, random_state=seed)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= self.arg.test_ratio / (self.arg.test_ratio + self.arg.validation_ratio), random_state=seed)
        return (np.float32(train_X), np.float32(train_y), np.float32(valid_X), np.float32(valid_y), np.float32(test_X), np.float32(test_y) )




##############################################################################################
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





##############################################################################################
class BaseModel(object):
    """
    This is BaseClass for model create

    Method:
        create_model : Initialize Model (torch.nn.Module)
        evaluate: 
        prepro_dataset:  (conver pandas.DataFrame to appropriate format)
        create_loss :   Initialize Loss Function 
        training:   starting training
        build: create model, loss, optimizer (call before training)
        train: equavilent to model.train() in pytorch (auto enable dropout,vv..vv..)
        eval: equavilent to model.eval() in pytorch (auto disable dropout,vv..vv..)
        device_setup: 
        load_DataFrame: read pandas
        load_weight: 
        save_weight: 
        predict : 
    """
    
    def __init__(self,arg):
        self.arg = Box(arg)
        self._device = self.device_setup(arg)
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
        self.criterior = self.create_loss().to(self.device)
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
            self.df = pd.read_csv(path,delimiter=';')
            return self.df
        if os.path.isfile(self.arg.DATASET.PATH):
            self.df = pd.read_csv(self.arg.DATASET.PATH,delimiter=';')
            return self.df
        else:
            import requests
            import io
            r = requests.get(self.arg.DATASET.URL)
            if r.status_code ==200:
                self.df = pd.read_csv(io.BytesIO(r.content),delimiter=';')
            else:
                raise Exception("Can't read data")
            
            return self.df


    def load_weight(self, path):
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
            
        
      torch.save(ckp,path)

    def predict(self,x,**kwargs):
        # raise NotImplementedError
        output = self.net(x,**kwargs)
        return output 






###################################################################################################
if __name__ == "__main__":
    import fire 
    fire.Fire()

