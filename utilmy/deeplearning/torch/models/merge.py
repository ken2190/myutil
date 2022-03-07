from turtle import forward
from base import BaseModel
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
class MergeEncoder_Create(BaseModel):
    """
    """
    def __init__(self,arg):
        super(MergeEncoder_Create,self).__init__(arg)
        
    def create_model(self,):
        # merge = self.arg.merge
        merge = getattr(self.arg.MODEL_INFO.MODEL_MERGE,'MERGE','add')
        skip = getattr(self.arg.MODEL_INFO.MODEL_MERGE,'SKIP',False)
        self.rule_encoder = RuleEncoder_Create(self.arg)
        self.data_encoder = DataEncoder_Create(self.arg)
        dims = self.arg.MODEL_INFO.MODEL_MERGE.ARCHITECT.decoder
        class Modelmerge(torch.nn.Module):
            def __init__(self,rule_encoder,data_encoder,alpha,dims,merge,skip):
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
        self.loss = self.create_loss().to(self.device)
        self.net = self.create_model().to(self.device)
        
    def create_loss(self,):
        rule_loss = self.rule_encoder.loss 
        data_loss = self.data_encoder.loss
        class MergeLoss(torch.nn.Module):

            def __init__(self,rule_loss,data_loss):
                super(MergeLoss,self).__init__(self)
                self.rule_loss = rule_loss
                self.data_loss = data_loss

            def forward(self,output,target,alpha):

                loss_1 = self.rule_loss(output,target)
                loss_2 = self.data_loss(output,target)                
                result = loss_1 * alpha + loss_2 * (1-alpha)
                return result

        return MergeLoss(rule_loss,data_loss)

    def load_DataFrame(self,path:str) ->pd.DataFame:
        pass

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
            rule_threshold = self.arg.rules.rule_threshold
            rule_ind       = self.arg.rules.rule_ind
            rule_feature   = 'ap_hi'
            src_unok_ratio = self.arg.rules.src_unok_ratio
            src_ok_ratio   = self.arg.rules.src_ok_ratio

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
        train_X, test_X, train_y, test_y = train_test_split(X_src,  y_src,  test_size=1 - arg.train_ratio, random_state=seed)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= arg.test_ratio / (arg.test_ratio + arg.validation_ratio), random_state=seed)
        return (train_X, train_y, valid_X,  valid_y, test_X,  test_y, )
        

def training(self,):
    df = self.load_DataFrame()
    train_X, train_y, valid_X,  valid_y, test_X,  test_y,  = self.prepro_dataset() 
    train_loader, valid_loader, test_loader = dataloader_create( train_X, train_y, valid_X, valid_y, test_X, test_y,  self.arg)

def model_train(model, losses, train_loader, valid_loader, arg:dict=None ):
    arg      = Box(arg)  ### Params
    arghisto = Box({})  ### results


    #### Rules Loss, params  ##################################################
    rule_feature   = arg.rules.get( 'rule_feature',   'ap_hi' )
    loss_rule_func = arg.rules.loss_rule_func   #ambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
    if 'loss_rule_calc' in arg.rules: loss_rule_calc = arg.rules.loss_rule_calc
    src_ok_ratio   = arg.rules.src_ok_ratio
    src_unok_ratio = arg.rules.src_unok_ratio
    rule_ind       = arg.rules.rule_ind
    pert_coeff     = arg.rules.pert_coeff


    #### Core model params
    model_params   = arg.model_info[ arg.model_type]
    lr             = model_params.get('lr',  0.001)
    optimizer      = torch.optim.Adam(model.parameters(), lr=lr)
    loss_task_func = losses.loss_task_func


    #### Train params
    model_type = arg.model_type
    # epochs     = arg.epochs
    early_stopping_thld    = arg.early_stopping_thld
    counter_early_stopping = 1
    # valid_freq     = arg.valid_freq
    seed=arg.seed
    log('saved_filename: {}\n'.format( arg.saved_filename))
    best_val_loss = float('inf')


    for epoch in range(1, arg.epochs+1):
      model.train()
      for batch_train_x, batch_train_y in train_loader:
        batch_train_y = batch_train_y.unsqueeze(-1)
        optimizer.zero_grad()

        if   model_type.startswith('dataonly'):  alpha = 0.0
        elif model_type.startswith('ruleonly'):  alpha = 1.0
        elif model_type.startswith('ours'):      alpha = arg.rules.alpha_dist.sample().item()
        arg.alpha = alpha

        ###### Base output #########################################
        output    = model(batch_train_x, alpha=alpha).view(batch_train_y.size())
        loss_task = loss_task_func(output, batch_train_y) #BCE


        ###### Loss Rule perturbed input and its output  #####################
        loss_rule = loss_rule_calc(model, batch_train_x, loss_rule_func, output, arg ) #F.mean()


        #### Total Losses  ##################################################
        scale = 1
        loss  = alpha * loss_rule + scale * (1 - alpha) * loss_task
        loss.backward()
        optimizer.step()


      # Evaluate on validation set
      if epoch % arg.valid_freq == 0:
        model.eval()
        if  model_type.startswith('ruleonly'):  alpha = 1.0
        else:                                   alpha = 0.0

        with torch.no_grad():
          for val_x, val_y in valid_loader:
            val_y = val_y.unsqueeze(-1)

            output = model(val_x, alpha=alpha).reshape(val_y.size())
            val_loss_task = loss_task_func(output, val_y).item()

            # perturbed input and its output
            pert_val_x = val_x.detach().clone()
            pert_val_x[:,rule_ind] = get_perturbed_input(pert_val_x[:,rule_ind], pert_coeff)
            pert_output = model(pert_val_x, alpha=alpha)    # \hat{y}_{p}    predicted sales from perturbed input

            val_loss_rule = loss_rule_func(output.reshape(pert_output.size()), pert_output).item()
            val_ratio = verification(pert_output, output, threshold=0.0).item()

            val_loss = val_loss_task

            y_true = val_y.cpu().numpy()
            y_score = output.cpu().numpy()
            y_pred = np.round(y_score)

            y_true = y_pred.reshape(y_true.shape[:-1])
            y_pred = y_pred.reshape(y_pred.shape[:-1])

            val_acc = mean_squared_error(y_true, y_pred)

          if val_loss < best_val_loss:
            counter_early_stopping = 1
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            log('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                  .format(epoch, best_val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio))
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, arg.saved_filename)
          else:
            log('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f}({}/{})'
                  .format(epoch, val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
            if counter_early_stopping >= early_stopping_thld:
              break
            else:
              counter_early_stopping += 1