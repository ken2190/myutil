from turtle import forward
from .base import BaseModel
import torch
import torch.nn as nn
from .ruleencoder import RuleEncoder_Create
from .dataencoder import DataEncoder_Create
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer,Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from .utils import dataloader_create
from utilmy import log, log2
from tqdm import tqdm
class MergeEncoder_Create(BaseModel):
    """
    """
    def __init__(self,arg):
        super(MergeEncoder_Create,self).__init__(arg)
        self.rule_encoder = RuleEncoder_Create(self.arg)
        self.data_encoder = DataEncoder_Create(self.arg)
    def create_model(self,):
        super(MergeEncoder_Create,self).create_model()
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
        log("rule_encoder:")
        self.rule_encoder.build()
        log("data_encoder:")
        self.data_encoder.build()
        log("MergeModel:")
        self.net = self.create_model().to(self.device)
        self.criterior = self.create_loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        
    def create_loss(self,):
        super(MergeEncoder_Create,self).create_loss()
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
        if df is not None:              
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
        

    def training(self,load_DataFrame=None,prepro_dataset=None):

        # training with load_DataFrame and prepro_data function or default funtion in self.method

        if load_DataFrame:
            self.load_DataFrame = load_DataFrame
        if prepro_dataset:
            self.prepro_dataset = prepro_dataset

        df = self.load_DataFrame()
        
        train_X, train_y, valid_X,  valid_y, test_X,  test_y, = self.prepro_dataset(df)
        train_loader, valid_loader, test_loader =  dataloader_create(train_X, train_y, 
                                                                    valid_X,  valid_y,
                                                                    test_X,  test_y,
                                                                    device=self.device,
                                                                    batch_size=self.arg.TRAINING_CONFIG.BATCH_SIZE)

        EPOCHS = self.arg.TRAINING_CONFIG.EPOCHS
        
        n_train = len(train_loader)
        n_val = len(valid_loader)
        
        for epoch in tqdm(range(1,EPOCHS+1)):
            self.train()
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
            self.eval()
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

