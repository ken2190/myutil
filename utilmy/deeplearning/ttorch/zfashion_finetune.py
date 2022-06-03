import warnings

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics

warnings.filterwarnings("ignore")

from nfnets import pretrained_nfnet

from torch.nn import functional as F


#######################################################################################################
#                                       NFNets Model                                                  #
#######################################################################################################

class myMultiClassMultiHead(nn.Module):
    def __init__(self, config, class_label_dict):
        super().__init__()
        self.config       = config
        self.in_features  = config['in_features']
        self.inter_feat   = config['inter_feat']
        self.class_label_dict  = class_label_dict
        self.dropout_prob = config['dropout']
        self.model_name   = config['model_name']

        if self.model_name == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
            # self.out_feature_size = self.effnet._conv_head.out_channels
        elif self.model_name == 'nfnets':
            self.model = pretrained_nfnet(config['model_path'])
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1] + [nn.AdaptiveMaxPool2d(1)]))
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

        # Layer 1
        self.linear1        = nn.Linear(in_features=self.in_features, out_features=256, bias=False)        
        # Layer 2
        self.linear2        = nn.Linear(in_features=256, out_features=self.inter_feat, bias=False)

        ########################################################################
        self.head_task_dict = {}
        for cls, lable in class_label_dict.items():
            self.head_task_dict[cls] = nn.Linear(self.inter_feat, lable)
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ  = torch.argmax(targ, dim=1).cpu().detach().numpy()
            #targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def forward(self, image,target_label_dict):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        targets = {}
        if 'gender' in target_label_dict:
            for cls,target in target_label_dict.items():
                targets[cls] = target
        else:
            targets = None
        out                   = {}
        for cls in target_label_dict.keys():
            out[cls] = self.head_task_dict[cls](x)

        if targets is not None:
            loss = []
            for k,v in out.items():
                loss.append(nn.CrossEntropyLoss()(out[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(out, targets)
            return out, loss, metrics
        return out, None, None
    
    def extract_features(self, image):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        return x



class FashionModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.config       = config
        self.in_features  = config['in_features']
        self.inter_feat   = config['inter_feat']
        self.num_classes  = num_classes
        self.dropout_prob = config['dropout']
        self.model_name   = config['model_name']



        if self.model_name == 'efficientnet':
            self.model = EfficientNet.from_pretrained('efficientnet-b3')
            # self.out_feature_size = self.effnet._conv_head.out_channels
        elif self.model_name == 'nfnets':
            self.model = pretrained_nfnet(config['model_path'])
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1] + [nn.AdaptiveMaxPool2d(1)]))
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

        # Layer 1
        self.linear1        = nn.Linear(in_features=self.in_features, out_features=256, bias=False)        
        # Layer 2
        self.linear2        = nn.Linear(in_features=256, out_features=self.inter_feat, bias=False)

        self.gender         = nn.Linear(self.inter_feat, self.num_classes['gender'])
        self.masterCategory = nn.Linear(self.inter_feat, self.num_classes['masterCategory'])
        self.subCategory    = nn.Linear(self.inter_feat, self.num_classes['subCategory'])
        self.articleType    = nn.Linear(self.inter_feat, self.num_classes['articleType'])
        self.baseColour     = nn.Linear(self.inter_feat, self.num_classes['baseColour'])
        self.season         = nn.Linear(self.inter_feat, self.num_classes['season'])
        self.usage          = nn.Linear(self.inter_feat, self.num_classes['usage'])
       
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, 
                articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        targets = {}
        if genderLabel is None:
            targets = None
        else:
            targets['gender']         = genderLabel
            targets['masterCategory'] = masterCategoryLabel
            targets['subCategory']    = subCategoryLabel
            targets['articleType']    = articleTypeLabel
            targets['baseColour']     = baseColourLabel
            targets['season']         = seasonLabel
            targets['usage']          = usageLabel
        out                   = {}
        out["gender"]         = self.gender(x)
        out["masterCategory"] = self.masterCategory(x)
        out["subCategory"]    = self.subCategory(x)
        out["articleType"]    = self.articleType(x)
        out["baseColour"]     = self.baseColour(x)
        out["season"]         = self.season(x)
        out["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in out.items():
                loss.append(nn.CrossEntropyLoss()(out[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(out, targets)
            return out, loss, metrics
        return out, None, None
    
    def extract_features(self, image):
        batch_size = image.shape[0]

        if self.model_name == 'nfnets':
            x = self.model(image).view(batch_size, -1)
        elif self.model_name == 'efficientnet':
            x = self.model.extract_features(image)
            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            
        else:
            x = image

        x = self.relu(self.linear1(self.dropout(x)))
        x = self.relu(self.linear2(self.dropout(x)))

        return x








import os
import albumentations
import  matplotlib.pyplot as plt
import pandas as pd

import tez

from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn

import torchvision

from sklearn import metrics, model_selection
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import argparse
import os

import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, model_selection, preprocessing
from tez.callbacks import EarlyStopping, Callback
from tez.datasets import ImageDataset
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import cv2

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

import json
import sys
from PIL import Image

from nfnets import pretrained_nfnet, SGD_AGC

# =====================================================================
# Dataset                                                        =
# =====================================================================
class FashionImageDataset(Dataset):

    def __init__(self, IMAGE_PATH, DF_PATH, labels_required = True, use_features=True, aug=None, inference=False):
        """
        Args:
            IMAGE_PATH (string): Directory with all the images or vectors
            DF_PATH (string): Path to csv file
            labels_required (bool): target labels required or not
            use_features (bool): set this to false if want to use images as source instead of vectors or use_features
            aug: augumentation
            inference: set it to True when use the model for inference
        """
        self.image_dir       = IMAGE_PATH
        self.df              = pd.read_csv(DF_PATH)
       
        self.labels_required = labels_required
        self.use_features    = use_features
        self.inference       = inference

        if self.use_features:
             self.images = [str(i) + '.npy' for i in self.df.id.tolist()][:100]
        else:
             self.images = [str(i) + '.jpg' for i in self.df.id.tolist()][:100]

        if aug is None:
          self.aug = albumentations.Compose(
              [
                  albumentations.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                      p=1.0,
                  ),
              ],
              p=1.0,
          )
        else:
          self.aug = aug

        if self.labels_required:
            self.genderLabels         = self.df.gender.tolist()
            self.masterCategoryLabels = self.df.masterCategory.tolist()
            self.subCategoryLabels    = self.df.subCategory.tolist()
            self.articleTypeLabels    = self.df.articleType.tolist()
            self.baseColourLabels     = self.df.baseColour.tolist()
            self.seasonLabels         = self.df.season.tolist()
            self.usageLabels          = self.df.usage.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename =self.images[idx]

        if self.use_features:
            img = np.load(os.path.join(self.image_dir, filename)).astype(np.float32)
            img = torch.from_numpy(img)
        else:
            # try:
            img = np.array(Image.open(os.path.join(self.image_dir, filename)).convert('RGB'))
            # img = cv2.imread(os.path.join(self.image_dir, filename))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(256,256))
            img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
            img = img.permute(2, 0, 1)
            # except:
            #     print(filename)
            #     img = torch.rand((3, 256, 256)).type(torch.FloatTensor)


        if self.labels_required:
            genderLabel         = torch.tensor(self.genderLabels[idx])
            masterCategoryLabel = torch.tensor(self.masterCategoryLabels[idx])
            subCategoryLabel    = torch.tensor(self.subCategoryLabels[idx])
            articleTypeLabel    = torch.tensor(self.articleTypeLabels[idx])
            baseColourLabel     = torch.tensor(self.baseColourLabels[idx])
            seasonLabel         = torch.tensor(self.seasonLabels[idx])
            usageLabel          = torch.tensor(self.usageLabels[idx])

            return {'image': img, 'genderLabel': genderLabel, 'masterCategoryLabel': masterCategoryLabel, 'subCategoryLabel': subCategoryLabel, 
                    'articleTypeLabel': articleTypeLabel, 'baseColourLabel': baseColourLabel, 'seasonLabel': seasonLabel, 'usageLabel': usageLabel
            }

        if self.inference:
            return {'image': img }

        return {'image': img, 'filename': filename.split('.')[0]}



# =====================================================================
# Dataset for Big Model1                                                 =
# =====================================================================


class FashionImageDatasetBigModel(Dataset):
            def __init__(self, IMAGE_PATH, DF_PATH, aug=None):
                """
                Args:
                    IMAGE_PATH (string): Directory with all the images or vectors
                    DF_PATH (string): Path to csv file
                    aug: augumentation
                    
                """
                self.image_dir = IMAGE_PATH
                self.df = pd.read_csv(DF_PATH)
            
        
                self.images = [str(i) + '.jpg' for i in self.df.id.tolist()]
                self.aug = aug
        
            def __len__(self):
                return len(self.images)
        
            def __getitem__(self, idx):
                filename =self.images[idx]
        
                # img = cv2.imread(os.path.join(self.image_dir, filename))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img,(256,256))
                # img = torch.from_numpy(self.aug(image=img)['image']).type(torch.FloatTensor)
                img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
                # print(np.array(img).shape, filename) 
                img = self.aug(img).type(torch.FloatTensor)
                # img = img.permute(2, 0, 1)
                # print(img.shape)
        
                return {'image': img, 'filename': filename.split('.')[0]}


# =====================================================================
# Feature Extractor Model                                                        =
# =====================================================================
class FeatureExtractorEfficientNet(nn.Module):
    def __init__(self, efficientnet_model_name='efficientnet-b3'):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained(efficientnet_model_name)
        self.out_feature_size = self.effnet._conv_head.out_channels

    def forward(self, image):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        return x


# =====================================================================
# Model    Small                                                    =
# =====================================================================
class FashionModel(tez.Model):
    def __init__(self, num_classes, in_features=1536, intermediate_features=512):
        super().__init__()

        # self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        # in_features = 1536
        # intermediate_features = 512
        # Layer 1
        self.bn1            = nn.BatchNorm1d(num_features=in_features)
        self.dropout1       = nn.Dropout(0.25)
        self.linear1        = nn.Linear(in_features=in_features, out_features=intermediate_features, bias=False)
        self.relu           = nn.ReLU()
        # Layer 2
        self.bn2            = nn.BatchNorm1d(num_features=intermediate_features)
        self.dropout2       = nn.Dropout(0.5)

        self.gender         = nn.Linear(intermediate_features, num_classes['gender'])
        self.masterCategory = nn.Linear(intermediate_features, num_classes['masterCategory'])
        self.subCategory    = nn.Linear(intermediate_features, num_classes['subCategory'])
        self.articleType    = nn.Linear(intermediate_features, num_classes['articleType'])
        self.baseColour     = nn.Linear(intermediate_features, num_classes['baseColour'])
        self.season         = nn.Linear(intermediate_features, num_classes['season'])
        self.usage          = nn.Linear(intermediate_features, num_classes['usage'])
       
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size, _ = image.shape

        x = self.linear1(self.dropout1(self.bn1(image)))
        x = self.relu(x)
        x = self.dropout2(self.bn2(x))

        targets = {}
        if genderLabel is None:
            targets = None
        else:
            targets['gender']         = genderLabel
            targets['masterCategory'] = masterCategoryLabel
            targets['subCategory']    = subCategoryLabel
            targets['articleType']    = articleTypeLabel
            targets['baseColour']     = baseColourLabel
            targets['season']         = seasonLabel
            targets['usage']          = usageLabel
        outputs                   = {}
        outputs["gender"]         = self.gender(x)
        outputs["masterCategory"] = self.masterCategory(x)
        outputs["subCategory"]    = self.subCategory(x)
        outputs["articleType"]    = self.articleType(x)
        outputs["baseColour"]     = self.baseColour(x)
        outputs["season"]         = self.season(x)
        outputs["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in outputs.items():
                loss.append(nn.CrossEntropyLoss()(outputs[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None
    
    def extract_features(self, image):
        batch_size, _ = image.shape

        features = self.linear1(self.dropout1(self.bn1(image)))

        return features
    


#######################################################################################################
#                                       NFNets Model                                                    #
#######################################################################################################


class FashionModelNFNets(nn.Module):
    def __init__(self, num_classes, in_features=3072, intermediate_features=512):
        super().__init__()
        self.model = pretrained_nfnet('pretrained/F0_haiku.npz')
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.pool = nn.AdaptiveMaxPool2d(1)

        # Layer 1
        self.dropout1       = nn.Dropout(0.25)
        self.linear1        = nn.Linear(in_features=in_features, out_features=intermediate_features, bias=False)
        self.relu           = nn.ReLU()
        
        # Layer 2
        self.dropout2       = nn.Dropout(0.2)

        self.gender         = nn.Linear(intermediate_features, num_classes['gender'])
        self.masterCategory = nn.Linear(intermediate_features, num_classes['masterCategory'])
        self.subCategory    = nn.Linear(intermediate_features, num_classes['subCategory'])
        self.articleType    = nn.Linear(intermediate_features, num_classes['articleType'])
        self.baseColour     = nn.Linear(intermediate_features, num_classes['baseColour'])
        self.season         = nn.Linear(intermediate_features, num_classes['season'])
        self.usage          = nn.Linear(intermediate_features, num_classes['usage'])
       
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        accuracy = []
        for k,v in outputs.items():
            out  = outputs[k]
            targ = targets[k]
            # print(out)
            out  = torch.argmax(out, dim=1).cpu().detach().numpy()
            targ = targ.cpu().detach().numpy()
            accuracy.append(metrics.accuracy_score(targ, out))
        return {'accuracy': sum(accuracy)/len(accuracy)}

    def forward(self, image, genderLabel=None, masterCategoryLabel=None, subCategoryLabel=None, articleTypeLabel=None, baseColourLabel=None, seasonLabel=None, usageLabel=None):
        batch_size = image.shape[0]

        x = self.pool(self.model(image)).view(batch_size, -1)


        x = self.linear1(self.dropout1(x))
        x = self.relu(x)
        x = self.dropout2(x)

        targets = {}
        if genderLabel is None:
            targets = None
        else:
            targets['gender']         = genderLabel
            targets['masterCategory'] = masterCategoryLabel
            targets['subCategory']    = subCategoryLabel
            targets['articleType']    = articleTypeLabel
            targets['baseColour']     = baseColourLabel
            targets['season']         = seasonLabel
            targets['usage']          = usageLabel
        outputs                   = {}
        outputs["gender"]         = self.gender(x)
        outputs["masterCategory"] = self.masterCategory(x)
        outputs["subCategory"]    = self.subCategory(x)
        outputs["articleType"]    = self.articleType(x)
        outputs["baseColour"]     = self.baseColour(x)
        outputs["season"]         = self.season(x)
        outputs["usage"]          = self.usage(x)

        if targets is not None:
            loss = []
            for k,v in outputs.items():
                loss.append(nn.CrossEntropyLoss()(outputs[k], targets[k]))
            loss = sum(loss)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None
    
    def extract_features(self, image):
        batch_size = image.shape[0]
        x = self.pool(self.model(image)).view(batch_size, -1)

        features = self.linear1(self.dropout1(x))

        return features



class SaveEfterEpoch(Callback):
    def __init__(self, model_path, model_name='model', save_interval=5):
        self.save_interval = save_interval
        self.model_path = model_path
        self.model_name = model_name
    
    def on_epoch_end(self, model):
        if model.current_epoch % self.save_interval == 0:
            model.save(os.path.join(self.model_path, f"{self.model_name}_epoch_{model.current_epoch}.pt"))
            print(f"model saved at epoch: {model.current_epoch}")


def encode_label(df, dfx, column):
    if column not in df.columns:
        print(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # print("Number of classes: ", len(le.classes_))
    return le.transform(dfx[column].tolist())

def decode_label(df, dfx, column):
    if column not in df.columns:
        print(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # print("Number of classes: ", len(le.classes_))
    return list(le.inverse_transform(dfx[column].tolist()))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path",                type=str,  default='./')
    parser.add_argument("--image_path",               type=str,  default='./data/fashion-dataset/images')
    parser.add_argument("--intermediate_vector_path", type=str,  default='./data/fashion-dataset/efficienet_vectors')
    parser.add_argument("--output_vector_path",        type=str,  default='./final_vectors')
    parser.add_argument("--model_name",               type=str,  default='fashion')
    parser.add_argument("--batch_size",               type=int,  default=8)
    parser.add_argument("--epochs",                   type=int,  default=5)
    parser.add_argument("--save_interval",            type=int,  default=2)
    parser.add_argument("--device",                   type=str,  default='cuda')
    parser.add_argument("--fp16",                     type=bool, default=False)
    parser.add_argument("--patience",                 type=int,  default=5)
    parser.add_argument("--input_vector_size",        type=int,  default=1536)
    parser.add_argument("--intermediate_vector_size", type=int,  default=512)
    parser.add_argument("--logistic_reg_targ_col",    type=str,  default='gender')
    parser.add_argument("--stage",                    type=str,  default='')
    parser.add_argument("--base_df",                  type=str,  default='')
    parser.add_argument("--root_df",                  type=str,  default='')
    parser.add_argument("--train_df",                 type=str,  default='')
    parser.add_argument("--val_df",                   type=str,  default='')
    parser.add_argument("--test_df",                  type=str,  default='')
    parser.add_argument("--pred_df",                  type=str,  default='')
    parser.add_argument("--image_ext",                type=str,  default='jpg')
    args = parser.parse_args()


    # args.stage = list(args.stage)
    print(args.stage)

    BASE_DIR                 = Path(args.base_path)
    IMAGE_PATH               = Path(args.image_path)
    INTERMEDIATE_VECTOR_PATH = Path(args.intermediate_vector_path)
    OUTPUT_VECTOR_PATH        = Path(args.output_vector_path)
    # BIG_MODEL_VECTOR_PATH    = Path(args.big_model_vector_path)

    MODEL_PATH               = BASE_DIR/'models'
    MODEL_NAME               = args.model_name
    TRAIN_BATCH_SIZE         = args.batch_size
    VALID_BATCH_SIZE         = args.batch_size
    EPOCHS                   = args.epochs

    
    os.makedirs(MODEL_PATH, exist_ok=True)



    print(f"Total Number of Images: ", len(list(IMAGE_PATH.glob(f"*.{args.image_ext}"))))


    ####### DF Preprocess  ##########################################################################################
    if 'df_preprocess' in args.stage:

        dfx = pd.read_csv(BASE_DIR/args.base_df, error_bad_lines=False)
        print("Dataset Size: ", dfx.shape)
        dfx = dfx.dropna()
        dfx = dfx[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]

        file_not_found = []
        for i in dfx.id.tolist():
            if not os.path.isfile(os.path.join(IMAGE_PATH, str(i) + f".{args.image_ext}")):
                file_not_found.append(i)
        print(f"Total {len(file_not_found)} images didn't found")

        
        dfx = dfx[~dfx['id'].isin(file_not_found)]
        dfx.to_csv(BASE_DIR/str(args.base_df.split('.')[0] + '_df.csv'), index=False)
        articleTypes = []
        for k, v in dict(dfx.articleType.value_counts()).items():
            if v <= 10:
                articleTypes.append(k)
        dfx = dfx[~dfx['articleType'].isin(articleTypes)]

        dfx.to_csv(BASE_DIR/(args.root_df.split('.')[0] + '_with_original_class.csv'), index=False)

        for col in dfx.columns.tolist()[1:]:
            dfx[col] = encode_label(dfx, dfx, col)
        
        print("Final Dataset Size: ", dfx.shape)
                
        dfx.to_csv(BASE_DIR/args.root_df, index=False)

        train_dfx, val_dfx = train_test_split(dfx, test_size=0.20, random_state=42)

        train_dfx.to_csv(BASE_DIR/args.train_df, index=False)
        val_dfx.to_csv(BASE_DIR/args.val_df, index=False)

        print("Stage: df_preprocess Finished ------------------------------------------------")

    # dfx = pd.read_csv(BASE_DIR/args.root_df)


    
    
    ####### Extract from Efficient Net  ##########################################################################################
    if 'extract_efficienet' in args.stage:
        os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        print("Starting Extract Vectors from Big model -------------------------")
        fashion_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/args.test_df, use_features=False, labels_required=False)
        print(len(fashion_dataset))
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

        model = FeatureExtractorEfficientNet()
        model.to(args.device)
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_VECTOR_PATH, f_name), image_vector[i])
                
                
        print("Stage: extract_efficienet Finished ------------------------------------------------")
        




    ####### Extract from igmodel  ##########################################################################################
    if 'extract_bigmodel1' in args.stage:
        # os.makedirs(INTERMEDIATE_VECTOR_PATH, exist_ok=True)
        os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        def get_features(model, dataloader, device='cuda'):
            all_features = []
            all_labels = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    features = model.encode_image(data['image'].to(device))
                    print(features.shape)
                    all_features.append(features)
                    all_labels.append(labels)
        
            return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
        
        os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        
        
        model = torch.jit.load(os.path.join(MODEL_PATH, args.model_name + '.pt')).cuda().eval()
        
        input_resolution = model.input_resolution.item()
        context_length   = model.context_length.item()
        vocab_size       = model.vocab_size.item()
        
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)
        
        
        preprocess = Compose([
            Resize(input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(input_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        
        fashion_dataset    = FashionImageDatasetBigModel(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/args.test_df, aug = preprocess)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        
        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vector = model.encode_image(data['image'].to(args.device)).cpu().detach().numpy()
                for i in range(len(data['filename'])):
                    f_name = data['filename'][i]
                    np.save(os.path.join(OUTPUT_VECTOR_PATH, f_name), image_vector[i])
        
        print("Stage extract_bigmodel1 Finished ------------------------------------------------")

    ##############################################################################################################    
    if 'train_small' in args.stage:
        # os.makedirs(INTERMEDIATE_VECTOR_PATH, exist_ok=True)
        # os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        print("Starting  Train Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        

        train_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH= BASE_DIR/args.train_df)
        val_dataset   = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH= BASE_DIR/args.val_df)
        

        ###### Model
        model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        
        es = EarlyStopping(
            monitor    = "valid_loss",
            model_path = os.path.join(MODEL_PATH, MODEL_NAME + "_best.pt"),
            patience   = args.patience,
            mode       = "min",
        )

        sfe = SaveEfterEpoch(save_interval=args.save_interval, model_path=MODEL_PATH, model_name=args.model_name)

        model.fit(
            train_dataset,
            valid_dataset = val_dataset,
            train_bs      = TRAIN_BATCH_SIZE,
            valid_bs      = VALID_BATCH_SIZE,
            device        = args.device,
            epochs        = EPOCHS,
            callbacks     = [es, sfe],
            fp16          = args.fp16,
        )

        model.save(os.path.join(MODEL_PATH, MODEL_NAME + "_last.pt"))
        print("Stage: train_small Finished -------------------------")


    ##################################################################################################################
    if 'extract_small' in args.stage:
        # os.makedirs(INTERMEDIATE_VECTOR_PATH, exist_ok=True)
        os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        print("Starting Extract Vector from Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/args.test_df, use_features=True, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

        model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))
        model.to(args.device)
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_VECTOR_PATH, f_name), image_vector[i])
        print("Stage: extract_small Finished -------------------------")



    ############ Logistic Regression ##################################################################################################    
    if 'check_vectors' in args.stage:
        """
           Load Final vectors, Load Labels (only one class), and train a logistic regression
           and check accuracy.
        
        
        """
        print("Starting   -------------------------", args.stage)
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        
        dfx_train = pd.read_csv(BASE_DIR/args.train_df)
        dfx_val = pd.read_csv(BASE_DIR/args.val_df)

        def create_logistic_regression_dataset(df, path=INTERMEDIATE_VECTOR_PATH, target_column=None, targets=None):
            X = []
            for i in df.id.tolist():
                X.append(np.load(os.path.join(path, str(i) + '.npy')).astype(np.float32))
            if targets is not None and target_column is not None:
                if target_column in df.columns.tolist():
                    y = np.array(df[target_column].tolist()).reshape(-1, 1)
                X = np.vstack(X)
            return X, y

        X_train, y_train = create_logistic_regression_dataset(dfx_train, INTERMEDIATE_VECTOR_PATH, target_column=args.logistic_reg_targ_col, targets=True)
        X_val, y_val = create_logistic_regression_dataset(dfx_train, INTERMEDIATE_VECTOR_PATH, target_column=args.logistic_reg_targ_col, targets=True)
        
        
        print(f"Traing Logistic Regression for {args.logistic_reg_targ_col}------------------")

        # # Perform logistic regression
        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
        classifier.fit(X_train, y_train)

        pickle.dump(classifier, open(os.path.join(MODEL_PATH, args.model_name + f"_{args.logistic_reg_targ_col}_.pkl"), 'wb'))
        
        # Evaluate using the logistic regression classifier
        predictions = classifier.predict(X_val)
        # print(predictions.shape)
        accuracy = np.mean((y_val.reshape(-1) == predictions.reshape(-1)).astype(np.float)) * 100.
        print(f"Accuracy = {accuracy:.3f}")
        # accuracy = accuracy_score(y_val.reshape(-1), predictions.reshape(-1))
        # print(f"Accuracy = {accuracy:.3f}")

        print("Stage: check_vectors Finished ------------------------------------------------")

    

    ######### Inference #####################################################################################################    
    if 'predict_small' in args.stage:

        print("Starting Extract Vector from Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        dfx_with_classes = pd.read_csv(BASE_DIR/args.base_df)
        test_df = pd.read_csv(BASE_DIR/args.test_df)

        
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INTERMEDIATE_VECTOR_PATH, DF_PATH=BASE_DIR/args.test_df, use_features=True, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))
        model.to(args.device)
        model.eval()


        pred_dict = dict([[key, []] for key in dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()]).keys()])
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vec = data['image'].to(args.device)
                filenames = data['filename']
                preds, _, _ = model(image_vec)
                pred_dict['id'].extend(data['filename'])
                for k,v in preds.items():
                    pred_dict[k].extend(torch.argmax(softmax(v), dim=1).cpu().detach().numpy())
        pred_df = pd.DataFrame.from_dict(pred_dict)

        print(pred_df.shape)
        if test_df.columns.tolist() == pred_df.columns.tolist():
            for col in dfx_with_classes.columns.tolist()[1:]:
                test_df[col] = encode_label(dfx_with_classes, test_df, col)
            for col in test_df.columns.tolist()[1:]:
                print(f"Class Name: {col} ========== Accuracy: {accuracy_score(test_df[col].values, pred_df[col].values):.3f}")


        for col in dfx.columns.tolist()[1:]:
            pred_df[col] = decode_label(dfx_with_classes, pred_df, col)

        pred_df.to_csv(BASE_DIR/args.pred_df, index=False)

        print("Prediction file saved to ", args.pred_df)
    
        print("Stage: predict_small Finished ------------------------------------------------")


 ######### NFNets Inference #####################################################################################################    
    if 'predict_nfnets' in args.stage:

        print("Starting Extract Vector from Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        dfx_with_classes = pd.read_csv(BASE_DIR/args.base_df)
        test_df = pd.read_csv(BASE_DIR/args.test_df)


        
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        fashion_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/args.test_df, use_features=False, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


        model = FashionModelNFNets(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        # model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        # model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))['state_dict'])
        model.to(args.device)
        model.eval()


        pred_dict = dict([[key, []] for key in dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()]).keys()])
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vec = data['image'].to(args.device)
                filenames = data['filename']
                preds, _, _ = model(image_vec)
                pred_dict['id'].extend(data['filename'])
                for k,v in preds.items():
                    pred_dict[k].extend(torch.argmax(softmax(v), dim=1).cpu().detach().numpy())
        pred_df = pd.DataFrame.from_dict(pred_dict)

        print(pred_df.shape)
        if test_df.columns.tolist() == pred_df.columns.tolist():
            for col in dfx_with_classes.columns.tolist()[1:]:
                test_df[col] = encode_label(dfx_with_classes, test_df, col)
            for col in test_df.columns.tolist()[1:]:
                print(f"Class Name: {col} ========== Accuracy: {accuracy_score(test_df[col].values, pred_df[col].values):.3f}")


        for col in dfx.columns.tolist()[1:]:
            pred_df[col] = decode_label(dfx_with_classes, pred_df, col)

        pred_df.to_csv(BASE_DIR/args.pred_df, index=False)

        print("Prediction file saved to ", args.pred_df)
    
        print("Stage: predict_small Finished ------------------------------------------------")


    ##################################################################################################################
    if 'extract_nfnets' in args.stage:
        # os.makedirs(INTERMEDIATE_VECTOR_PATH, exist_ok=True)
        os.makedirs(OUTPUT_VECTOR_PATH, exist_ok=True)
        print("Starting Extract Vector from Small model -------------------------")
        dfx = pd.read_csv(BASE_DIR/args.root_df)
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
        fashion_dataset = FashionImageDataset(IMAGE_PATH=IMAGE_PATH, DF_PATH=BASE_DIR/args.test_df, use_features=False, labels_required=False)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

        # model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        # model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))
        # model.to(args.device)
        # model.eval()

        model = FashionModelNFNets(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        # model = FashionModel(num_classes=class_dict, in_features=args.input_vector_size, intermediate_features=args.intermediate_vector_size)
        # model.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_NAME + ".pt"))['state_dict'])
        model.to(args.device)
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(args.device)).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_VECTOR_PATH, f_name), image_vector[i])
        print("Stage: extract_small Finished -------------------------")



        



        
        





 







































import argparse
import os
import warnings
from pathlib import Path

import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from efficientnet_pytorch import EfficientNet, EfficientNet
from sklearn import metrics, metrics, model_selection, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from tez.callbacks import Callback, EarlyStopping, EarlyStopping
from tez.datasets import ImageDataset, ImageDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


from Path import  pathlib
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

import json
import sys
from PIL import Image

from nfnets import pretrained_nfnet, SGD_AGC

import deepspeed

from torch.nn import functional as F

from trainer import FashionModel


def log(*s):
    print(*s, flush=True)


def log2(*s):
    print(*s, flush=True)

def logw(*s):
    print(*s, flush=True)


def loge(*s):
    print(*s, flush=True)





from datasets import FashionImageDataset

from models import FashionModel



def encode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return le.transform(dfx[column].tolist())


def decode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return list(le.inverse_transform(dfx[column].tolist()))



def config_get_paths(cfg, m_config=None, name='preprocess/train/predict'):

    if 'preprocess' in name :
        config_preprocess = cfg['preprocess']
        IMAGE_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])

        return IMAGE_PATH, CSV_PATH


    if 'train' in name or 'predict' in name:
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
        MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                                  m_config['model_name'], str(m_config['version']))

        if cfg['feature_extractor']['input_data_type'] == 'train':
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

        else:
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])


        log(INPUT_PATH)
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        return   CSV_PATH, MODEL_PATH, INPUT_PATH, OUTPUT_PATH



def generate_dataset(config):
    
    cfg2 = config['preprocess']

    IMAGE_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    os.makedirs(CSV_PATH, exist_ok=True)
    
    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg2['base_df']), error_bad_lines=False)
    log("Dataset Size: ", dfx.shape)
    dfx = dfx.dropna()
    dfx = dfx[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]

    file_not_found = []
    for i in dfx.id.tolist():
        flag = False
        for ext in cfg2['image_exts']:
            if os.path.isfile(os.path.join(IMAGE_PATH, str(i) + f".{ext}")):
                flag = True
                break
        if not flag:
            file_not_found.append(i)
    log(f"Total {len(file_not_found)} images didn't found")

    
    dfx = dfx[~dfx['id'].isin(file_not_found)]
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['base_df'].split('.')[0] + '_df.csv'), index=False)
    articleTypes = []
    for k, v in dict(dfx.articleType.value_counts()).items():
        if v <= 10:
            articleTypes.append(k)
    dfx = dfx[~dfx['articleType'].isin(articleTypes)]

    # dfx.to_csv(config_preprocess['preprocessed_df_with_classes'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df_with_classes']), index=False)

    for col in dfx.columns.tolist()[1:]:
        dfx[col] = encode_label(dfx, dfx, col)
    
    log("Final Dataset Size: ", dfx.shape)
            
    # dfx.to_csv(config_preprocess['preprocessed_df'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df']), index=False)
    train_dfx, val_dfx = train_test_split(dfx, test_size=cfg2["val_set_size"], random_state=42)

    # train_dfx.to_csv(config_preprocess['train_df'], index=False)
    # val_dfx.to_csv(config_preprocess['val_df'], index=False)
    train_dfx.to_csv(os.path.join(CSV_PATH, cfg2['train_df']), index=False)
    val_dfx.to_csv(os.path.join(CSV_PATH, cfg2['val_df']), index=False)
    log("Stage: df_preprocess Finished ------------------------------------------------")















import argparse
import os
import warnings
from pathlib import Path

import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from efficientnet_pytorch import EfficientNet, EfficientNet
from sklearn import metrics, metrics, model_selection, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from tez.callbacks import Callback, EarlyStopping, EarlyStopping
from tez.datasets import ImageDataset, ImageDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle

import json
import sys
from PIL import Image

from nfnets import pretrained_nfnet, SGD_AGC

import deepspeed

from torch.nn import functional as F

from trainer import FashionModel


def log(*s):
    print(*s, flush=True)































##########################################################################################
################### donwload  ############################################################




# =====================================================================
# Dataset for Big Model1                                                =
# =====================================================================
from datasets import FashionImageDataset




def encode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return le.transform(dfx[column].tolist())


def decode_label(df, dfx, column):
    if column not in df.columns:
        log(f"column: {column} nor present in dataframe")
    le = preprocessing.LabelEncoder()
    le.fit(df[column].tolist())
    # log("Number of classes: ", len(le.classes_))
    return list(le.inverse_transform(dfx[column].tolist()))



def config_get_paths(cfg, m_config=None, name='preprocess/train/predict'):

    if 'preprocess' in name :
        config_preprocess = cfg['preprocess']
        IMAGE_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])

        return IMAGE_PATH, CSV_PATH


    if 'train' in name or 'predict' in name:
        CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
        MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                                  m_config['model_name'], str(m_config['version']))

        if cfg['feature_extractor']['input_data_type'] == 'train':
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

        else:
            if m_config['model_name'] == 'smallmodel':
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
            else:
                INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])
            OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])


        log(INPUT_PATH)
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        return   CSV_PATH, MODEL_PATH, INPUT_PATH, OUTPUT_PATH



def generate_dataset(config):
    
    cfg2 = config['preprocess']

    IMAGE_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['train_dir'])

    CSV_PATH = os.path.join(config['base_dir'], config['input_dir'], config['dataset_name'], config['csv_dir'])

    os.makedirs(CSV_PATH, exist_ok=True)
    
    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg2['base_df']), error_bad_lines=False)
    log("Dataset Size: ", dfx.shape)
    dfx = dfx.dropna()
    dfx = dfx[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]

    file_not_found = []
    for i in dfx.id.tolist():
        flag = False
        for ext in cfg2['image_exts']:
            if os.path.isfile(os.path.join(IMAGE_PATH, str(i) + f".{ext}")):
                flag = True
                break
        if not flag:
            file_not_found.append(i)
    log(f"Total {len(file_not_found)} images didn't found")

    
    dfx = dfx[~dfx['id'].isin(file_not_found)]
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['base_df'].split('.')[0] + '_df.csv'), index=False)
    articleTypes = []
    for k, v in dict(dfx.articleType.value_counts()).items():
        if v <= 10:
            articleTypes.append(k)
    dfx = dfx[~dfx['articleType'].isin(articleTypes)]

    # dfx.to_csv(config_preprocess['preprocessed_df_with_classes'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df_with_classes']), index=False)

    for col in dfx.columns.tolist()[1:]:
        dfx[col] = encode_label(dfx, dfx, col)
    
    log("Final Dataset Size: ", dfx.shape)
            
    # dfx.to_csv(config_preprocess['preprocessed_df'], index=False)
    dfx.to_csv(os.path.join(CSV_PATH, cfg2['preprocessed_df']), index=False)
    train_dfx, val_dfx = train_test_split(dfx, test_size=cfg2["val_set_size"], random_state=42)

    # train_dfx.to_csv(config_preprocess['train_df'], index=False)
    # val_dfx.to_csv(config_preprocess['val_df'], index=False)
    train_dfx.to_csv(os.path.join(CSV_PATH, cfg2['train_df']), index=False)
    val_dfx.to_csv(os.path.join(CSV_PATH, cfg2['val_df']), index=False)
    log("Stage: df_preprocess Finished ------------------------------------------------")




def feature_extractor(cfg, m_config):

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                              m_config['model_name'], str(m_config['version']))

    if cfg['feature_extractor']['input_data_type'] == 'train':
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])

        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

    else:
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])

    log(INPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)


    log("########### Feature Extraction Started ############################################")
    if m_config['model_name'] == 'bigmodel':

        MODEL_PATH = m_config["model_path"]

        def get_features(model, dataloader, device='cuda'):
            all_features = []
            all_labels = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(dataloader)):
                    features = model.encode_image(data['image'].to(device))
                    log(features.shape)
                    all_features.append(features)
                    all_labels.append(labels)
        
            return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

        dfx = pd.read_csv(os.path.join(CSV_PATH, cfg['root_df']))
        
        model = torch.jit.load(MODEL_PATH).to(cfg["device"]).eval()
        
        input_resolution = model.input_resolution.item()
        context_length   = model.context_length.item()
        vocab_size       = model.vocab_size.item()
        
        log("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        log("Input resolution:", input_resolution)
        log("Context length:", context_length)
        log("Vocab size:", vocab_size)
        
        
        preprocess = Compose([
            Resize(input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(input_resolution),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        
        fashion_dataset    = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config, aug = preprocess)
        fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)
        
        
        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vector = model.encode_image(data['image'].to(cfg["device"])).cpu().detach().numpy()
                for i in range(len(data['filename'])):
                    f_name = data['filename'][i]
                    np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    else:

        MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

        dfx = pd.read_csv(os.path.join(CSV_PATH, cfg['root_df']))
        class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

        if m_config['model_name'] in ['efficientnet', 'nfnets']:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)
        else:
            fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)

        fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)

        model = FashionModel(config=m_config, num_classes=class_dict)
        model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
        model.to(cfg["device"])
        model.eval()

        for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
            image_vector = model.extract_features(data['image'].to(cfg["device"])).cpu().detach().numpy()
            for i in range(len(data['filename'])):
                f_name = data['filename'][i]
                np.save(os.path.join(OUTPUT_PATH, f_name), image_vector[i])
    log("########### Feature Extraction Finished ###########")



def inference(cfg, m_config):

    log("########### Prediction Started ###########")

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])
    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], \
                              m_config['model_name'], str(m_config['version']))
    if cfg['feature_extractor']['input_data_type'] == 'train':
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['train_dir'])
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['output_dir'], cfg['dataset_name'], m_config['model_name'])
    else:
        if m_config['model_name'] == 'smallmodel':
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], "test_" + m_config['input_data_dir'])
        else:
            INPUT_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['test_dir'])
        OUTPUT_PATH = os.path.join(cfg['base_dir'], cfg['output_dir'], cfg['dataset_name'], "test_" + m_config['model_name'])

    log(INPUT_PATH)


    os.makedirs(OUTPUT_PATH, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_PATH, 'checkpoint_best.pt')

    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg["preprocess"]["preprocessed_df_with_classes"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])

    if m_config['model_name'] in ['efficientnet', 'nfnets']:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)
    else:
        fashion_dataset = FashionImageDataset(IMAGE_PATH=INPUT_PATH, config=cfg, m_config=m_config)

    fashion_dataloader = DataLoader(fashion_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8)

    model = FashionModel(config=m_config, num_classes=class_dict)
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    model.to(cfg["device"])
    model.eval()


    with torch.no_grad():
        pred_dict = dict([[key, []] for key in dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()]).keys()])
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, data in tqdm(enumerate(fashion_dataloader), total=len(fashion_dataloader)):
                image_vec = data['image'].to(cfg["device"])
                filenames = data['filename']
                preds, _, _ = model(image_vec)
                pred_dict['id'].extend(data['filename'])
                for k,v in preds.items():
                    pred_dict[k].extend(torch.argmax(softmax(v), dim=1).cpu().detach().numpy())
        pred_df = pd.DataFrame.from_dict(pred_dict)

    log(pred_df.shape)

    for col in dfx.columns.tolist()[1:]:
        pred_df[col] = decode_label(dfx, pred_df, col)

    out = os.path.join(OUTPUT_PATH, m_config["model_name"] + "_" + str(m_config["version"]) + "_predicted.csv")
    pred_df.to_csv(out, index=False)

    log("Prediction file saved to ", out)

    log("########### Prediction Finished ###########")

























def logistic_regression(cfg, m_config):

    """
           Load Final vectors, Load Labels (only one class), and train a logistic regression
           and check accuracy.
        
        
    """

    log("########### Prediction Started ###########")

    CSV_PATH = os.path.join(cfg['base_dir'], cfg['input_dir'], cfg['dataset_name'], cfg['csv_dir'])

    MODEL_PATH = os.path.join(cfg['base_dir'], cfg['models_dir'], cfg['dataset_name'], 'logistic')

    INPUT_PATH = os.path.join(cfg['base_dir'], cfg['vector_dir'], cfg['dataset_name'], m_config['model_name'])

    TARGET_COL = cfg["logistic_regression"]["target_col"]


    os.makedirs(MODEL_PATH, exist_ok=True)

    dfx = pd.read_csv(os.path.join(CSV_PATH, cfg["root_df"]))
    class_dict = dict([[i, dfx[i].nunique()] for i in dfx.columns.tolist()[1:]])
    
    dfx_train = pd.read_csv(os.path.join(CSV_PATH, cfg["train_df"]))
    dfx_val = pd.read_csv(os.path.join(CSV_PATH, cfg["val_df"]))

    def create_logistic_regression_dataset(df, path=INPUT_PATH, target_column=None, targets=None):
        X = []
        for i in df.id.tolist():
            X.append(np.load(os.path.join(path, str(i) + '.npy')).astype(np.float32))
        if targets is not None and target_column is not None:
            if target_column in df.columns.tolist():
                y = np.array(df[target_column].tolist()).reshape(-1, 1)
            X = np.vstack(X)
        return X, y

    X_train, y_train = create_logistic_regression_dataset(dfx_train, INPUT_PATH, target_column=TARGET_COL, targets=True)
    X_val, y_val = create_logistic_regression_dataset(dfx_train, INPUT_PATH, target_column=TARGET_COL, targets=True)
    
    
    log(f"Traing Logistic Regression for {TARGET_COL}------------------")

    # # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=-1)
    classifier.fit(X_train, y_train)

    pickle.dump(classifier, open(os.path.join(MODEL_PATH, f'{m_config["model_name"]}_{m_config["version"]}_{TARGET_COL}.pkl'), 'wb'))
    
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(X_val)
    # log(predictions.shape)
    accuracy = np.mean((y_val.reshape(-1) == predictions.reshape(-1)).astype(np.float)) * 100.
    log(f"Accuracy = {accuracy:.3f}")
    # accuracy = accuracy_score(y_val.reshape(-1), predictions.reshape(-1))
    # log(f"Accuracy = {accuracy:.3f}")

    log("Logistic Regression Finished ------------------------------------------------")




def add_argument():

    parser = argparse.ArgumentParser(description='FASHION')

    parser.add_argument('--cfg',
                        type=str,
                        default="./cfg.json",
                        help='cconfig file')

    parser.add_argument('--stage',
                        type=str,
                        required=True,
                        help='stage')
    parser.add_argument('--model',
                        type=str,
                        default="bigmodel",
                        help='stage')
    args = parser.parse_args()
    return args



def run(cfg='./confing.json', stage='train', model='bigmodel'):
    """
       python  utils.py  run  --stage preprocess  --model bigmodel


    """
    #args = add_argument()

    with open(cfg) as f:
            config = json.load(f)

    if stage == "preprocess":
        generate_dataset(config)

    elif stage == "feature_extract":
        m_config = config["model_params"][model]
        log(m_config)
        feature_extractor(config, m_config)

    elif stage == "inference":
        m_config = config["model_params"][model]
        log(m_config)
        inference(config, m_config)

    elif stage == "logistic":
        m_config = config["model_params"][model]
        log(m_config)
        logistic_regression(config, m_config)



######################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()














if __name__ == "__main__":
   import fire
   fire.Fire()


