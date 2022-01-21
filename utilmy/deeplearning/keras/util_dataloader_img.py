# -*- coding: utf-8 -*-
HELP = """
 utils keras for dataloading
"""
import os,io, numpy as np, sys, glob, time, copy, json, pandas as pd, functools, sys
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence  



######################################################################################
import cv2
# import tifffile.tifffile
# from skimage import morphology
import PIL
from PIL import Image
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Resize
)
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch import batch_norm




###################################################################################################
from utilmy import log, log2

def help():
    from utilmy import help_create
    ss = HELP + help_create("utilmy.deeplearning.keras.util_dataloader_img")
    print(ss)


###################################################################################################    
def test():    
  #image_size = 64
  train_augments = Compose([
      #Resize(image_size, image_size, p=1),
      HorizontalFlip(p=0.5),
      #RandomContrast(limit=0.2, p=0.5),
      #RandomGamma(gamma_limit=(80, 120), p=0.5),
      #RandomBrightness(limit=0.2, p=0.5),
      #HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
      #                   val_shift_limit=10, p=.9),
      ShiftScaleRotate(
          shift_limit=0.0625, scale_limit=0.1, 
          rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
      # ToFloat(max_value=255),
      Transform_sprinkle(p=0.5),
  ])

  test_augments = Compose([
      # Resize(image_size, image_size, p=1),
      # ToFloat(max_value=255)
  ])

  ###############################################################################
  image_size =  64
  train_transforms = Compose([
      Resize(image_size, image_size, p=1),
      HorizontalFlip(p=0.5),
      RandomContrast(limit=0.2, p=0.5),
      RandomGamma(gamma_limit=(80, 120), p=0.5),
      RandomBrightness(limit=0.2, p=0.5),
      HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                         val_shift_limit=10, p=.9),
      ShiftScaleRotate(
          shift_limit=0.0625, scale_limit=0.1, 
          rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
      ToFloat(max_value=255),
      Transform_sprinkle(num_holes=10, side_length=10, p=0.5),
  ])

  test_transforms = Compose([
      Resize(image_size, image_size, p=1),
      ToFloat(max_value=255)
  ])
  
  
def test1():
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()

    train_loader = DataGenerator_img(X_train, y_train)
    valid_loader = DataGenerator_img(X_valid, y_valid)

    for i, (image, label) in enumerate(train_loader):
        print('Training : ')
        print(f'image shape : {image.shape}')
        print(f'label shape : {label.shape}')
        break

    for i, (image, label) in enumerate(valid_loader):
        print('\nValidation : ')
        print(f'image shape : {image.shape}')
        print(f'label shape : {label.shape}')
        break


def test2(): #using predefined df
    from numpy import random
    from pathlib import Path

    folder_name = 'random images'
    csv_file_name = 'df.csv'
    p = Path(folder_name)
    num_images = 50

    num_labels = 2
    
    def create_random_images_ds(img_shape, num_images = 10, folder = 'random images', return_df = True, num_labels = 2, label_cols = ['label']):
        if not os.path.exists(folder):
            os.mkdir(folder)
        for n in range(num_images):
            filename = f'{folder}/{n}.jpg'
            rgb_img = numpy.random.rand(img_shape[0],img_shape[1],img_shape[2]) * 255
            image = Image.fromarray(rgb_img.astype('uint8')).convert('RGB')
            image.save(filename)

        label_dict = []

        files = [i.as_posix() for i in p.glob('*.jpg')]
        for i in enumerate(label_cols):
            label_dict.append(random.randint(num_labels, size=(num_images)))

        zipped = list(zip(files, *label_dict))
        df = pd.DataFrame(zipped, columns=['uri'] + label_cols)
        if return_df:
            return df

    df = create_random_images_ds((28, 28, 3), num_images = num_images, num_labels = num_labels, folder = folder_name)
    df.to_csv(csv_file_name, index=False)

    dt_loader = DataGenerator_img_disk(p.as_posix(), df, ['label'], batch_size = 32)

    for i, (image, label) in enumerate(dt_loader):
        print(f'image shape : {(image).shape}')
        print(f'label shape : {(label).shape}')
        break


 
################################################################################################## 
##################################################################################################
def get_data_sample(batch_size, x_train, labels_val, labels_col):   #name changed
    """ Get a data sample X, Y_multilabel, with batch size from dataset
    Args:
        batch_size (int): Provide a batch size for sampling
        x_train (list): Inputs from the dataset
        labels_val (list): True labels for the dataset
        labels_col(list): Samples to select from these columns

    Returns:
        x (numpy array): Selected samples of size batch_size
        y_label_list (list): List of labels from selected samples  
        
    """
    #### 
    # i_select = 10
    # i_select = np.random.choice(np.arange(train_size), size=batch_size, replace=False)
    col0 = labels_col[0]
    i_select = np.random.choice(np.arange(len(labels_val[ col0 ])), size=batch_size, replace=False)

    #### Images
    x        = np.array([ x_train[i]  for i in i_select ] )

    #### y_onehot Labels  [y1, y2, y3, y4]
    # labels_col   = [  'gender', 'masterCategory', 'subCategory', 'articleType' ] #*To make user-defined
    y_label_list = []
    for ci in labels_col :
        v =  labels_val[ci][i_select]
        y_label_list.append(v)

    return x, y_label_list 


def pd_get_onehot_dict(df, labels_col:list, dfref=None, ) :       #name changed
    """
    Args:
        df (DataFrame): Actual DataFrame
        dfref (DataFrame): Reference DataFrame 
        labels_col (list): List of label columns

    Returns:
        dictionary: label_columns, count
    """
    if dfref is not None :
        df       = df.merge(dfref, on = 'id', how='left')
    
    labels_val = {}
    labels_cnt = {}
    for ci in labels_col:
      dfi_1hot  = pd.get_dummies(df, columns=[ci])  ### OneHot
      dfi_1hot  = dfi_1hot[[ t for t in dfi_1hot.columns if ci in t   ]].values  ## remove no OneHot
      labels_val[ci] = dfi_1hot 
      labels_cnt[ci] = df[ci].nunique()
      assert dfi_1hot.shape[1] == labels_cnt[ci],   labels_cnt     
    
    print(labels_cnt)
    return labels_val, labels_cnt
    
    

def pd_merge_labels_imgdir(dflabels, img_dir="*.jpg", labels_col = []) :   #name changed
    """One Hot encode label_cols
    #    id, uri, cat1, cat2, .... , cat1_onehot
    Args:
        dflabels (DataFrame): DataFrame to perform one hot encoding on
        img_dir (Path(str)): String Path /*.png to image directory
        labels_col (list): Columns to perform One Hot encoding on. Defaults to []

    Returns:
        DataFrame: One Hot encoded DataFrame
    """
    import glob
    fpaths   = glob.glob(img_dir )
    fpaths   = [ fi for fi in fpaths if "." in fi.split("/")[-1] ]
    log(str(fpaths)[:100])

    df         = pd.DataFrame(fpaths, columns=['uri'])
    log(df.head(1).T)
    df['id']   = df['uri'].apply(lambda x : x.split("/")[-1].split(".")[0]    )
    # df['id']   = df['id'].apply( lambda x: int(x) )
    df         = df.merge(dflabels, on='id', how='left')

    # labels_col = [  'gender', 'masterCategory', 'subCategory', 'articleType' ]
    for ci in labels_col :
      dfi_1hot           = pd.get_dummies(df, columns=[ci])  ### OneHot
      dfi_1hot           = dfi_1hot[[ t for t in dfi_1hot.columns if ci in t   ]]  ## keep only OneHot
      df[ci + "_onehot"] = dfi_1hot.apply( lambda x : ','.join([   str(t) for t in x  ]), axis=1)
      #####  0,0,1,0 format   log(dfi_1hot)
    return df


def pd_to_onehot(dflabels, labels_col = []) :   #name changed
    """One Hot encode label_cols for predefined df
    #    id, uri, cat1, cat2, .... , cat1_onehot
    Args:
        dflabels (DataFrame): DataFrame to perform one hot encoding on
        labels_col (list): Columns to perform One Hot encoding on. Defaults to []

    Returns:
        DataFrame: One Hot encoded DataFrame
    """
    for ci in labels_col :
      dfi_1hot           = pd.get_dummies(dflabels, columns=[ci])  ### OneHot
      dfi_1hot           = dfi_1hot[[ t for t in dfi_1hot.columns if ci in t   ]]  ## keep only OneHot
      dflabels[ci + "_onehot"] = dfi_1hot.apply(lambda x : ','.join([str(t) for t in x]), axis=1)
      #####  0,0,1,0 format   log(dfi_1hot)

    return dflabels




#################################################################################      
class DataGenerator_img(Sequence):
    """Custom DataGenerator using keras Sequence for image data in numpy array
    Args:
        x (np array): The input samples from the dataset
        y (np array): The labels from the dataset
        batch_size (int, optional): batch size for the samples. Defaults to 32.
        augmentations (str, optional): perform augmentations to the input samples. Defaults to None.
    """
    
    def __init__(self, x, y, batch_size=32, augmentations=None):
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.augment    = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for y_head in self.y:                                                         ----
        #     batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size]) ----
        if self.augment is not None:
            batch_x = np.stack([self.augment(image=x)['image'] for x in batch_x], axis=0)
        # return (batch_x, *batch_y)                                                    ----
        return (batch_x, batch_y)

from keras_preprocessing.image import ImageDataGenerator
from utilmy import pd_read_file
import math

class DataGenerator_img_disk:
    """
        Custom Image Data Loader and Augmention Generator,
        that wrappes keras API, and can be use to load dataset from 
        a directory class orgnized dataset or from a labeled csv file

        # Arguments
            image_dir: string, path to image directory
            label_path: string, path to label csv file, 
                        provide None if dataset is folder orgnized.
            label_cols: list, ['image path column name', 'image label column name'],
                        Only relavent if label_path is provided.
            validate_filenames: bool, if True then ignore if image is not found,
                        Only relavent if label_path is provided.
            batch_size: int, size of the batches of data (default: 8).
            class_mode: string, One of "categorical", "binary", "sparse",
                        "input", or None. Default: "categorical".
                        Determines the type of label arrays that are returned:
                        - "categorical" will be 2D one-hot encoded labels,
                        - "binary" will be 1D binary labels,
                            "sparse" will be 1D integer labels,
                        - "input" will be images identical
                            to input images (mainly used to work with autoencoders).
                        - If None, no labels are returned
                        (the generator will only yield batches of image data,
                        which is useful to use with `model.predict_generator()`).
                        Please note that in case of class_mode None,
                        the data still needs to reside in a subdirectory
                        of `directory` for it to work correctly.
            classes: list of class subdirectories
                    (e.g. `['dogs', 'cats']`). Default: None.
                    If not provided, the list of classes will be automatically
                    inferred from the subdirectory names/structure
                    under `directory`, where each subdirectory will
                    be treated as a different class
                    (and the order of the classes, which will map to the label
                    indices, will be alphanumeric).
            imgs_target_config: dictionary, 
                                'target_size':(256, 256), output target image size
                                'color_mode':'rgb', output target image mode,
                                        One of "grayscale", "rgb", "rgba". Default: "rgb".
                                        Whether the images will be converted to
                                        have 1, 3, or 4 channels.
                                'interpolation': 'nearest', what strategy to use when upsampling a low resolution image,
                                        Interpolation method used to resample the image if the
                                        target size is different from that of the loaded image.
                                        Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                                        If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                                        supported. If PIL version 3.4.0 or newer is installed,
                                        `"box"` and `"hamming"` are also supported.
                                        By default, `"nearest"` is used.
            transforms: dict, optional dictionary for image augmentations,
                        doc can be found at 'https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator'
            split_config: dict, optional.
                            'split_type':'training', (`training` or `validation`) 
                            'validation_split':0.0, Float. Fraction of images reserved for validation (strictly between 0 and 1).
            shuffle_config: dict, optional.
                            'shuffle': True, shuffle dataset batchs or not
                            'seed': None, seed for the shuffling
            save_config: dict, optional.
                            'save_to_dir': None, directory to save augmented images
                            'save_format': 'png', save format of images
    """

    generator = None
    
    transforms = {
        'featurewise_center': False,
        'samplewise_center': False,
        'featurewise_std_normalization': False,
        'samplewise_std_normalization': False,
        'zca_whitening': False,
        'zca_epsilon': 1e-6,
        'rotation_range': 0,
        'width_shift_range': 0.,
        'height_shift_range': 0.,
        'brightness_range': None,
        'shear_range': 0.,
        'zoom_range': 0.,
        'channel_shift_range': 0.,
        'fill_mode': 'nearest',
        'cval': 0.,
        'horizontal_flip': False,
        'vertical_flip': False,
        'rescale': None,
        'preprocessing_function': None,
        'data_format': 'channels_last',
        'interpolation_order': 1,
        'dtype': 'float32'
    }
    
    imgs_target_config = {
        'target_size':(256, 256), 
        'color_mode':'rgb', 
        'interpolation': 'nearest'
    }

    def __init__(self, img_dir, label_dir=None, label_cols:list=None, validate_filenames:bool=True, batch_size=8, class_mode:str='categorical', 
                 classes:list=None, *,
                imgs_target_config: dict={'target_size':(256, 256), 'color_mode':'rgb', 'interpolation': 'nearest'},
                transforms:dict=None,
                split_config:dict={'split_type':'training', 'validation_split':0.0},
                shuffle_config:dict={'shuffle': True, 'seed': None},
                save_config:dict={'save_to_dir': None, 'save_format': 'png'}) -> None:

        self.img_dir = img_dir # parent Images directory
        self.label_dir = label_dir # csv File that contains labels
        self.label_cols = label_cols # ['image path column name', 'image label']
        self.validate_filenames = validate_filenames # if csv file is probided, if True then ignore if image is not found
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.classes = classes
        
        # filter only what the user have provided and set the reset to the default values
        intersected_keys = set(imgs_target_config.keys()) & set(self.imgs_target_config.keys())
        for key in list(intersected_keys):
            self.imgs_target_config[key] = imgs_target_config[key]

        # filter only what the user have provided and set the reset to the default values
        if transforms:
            intersected_keys = set(transforms.keys()) & set(self.transforms.keys())
            for key in list(intersected_keys):
                self.transforms[key] = transforms[key]
            
        self.split_config = split_config
        self.shuffle_config = shuffle_config
        self.save_config = save_config

        self.data_gen = ImageDataGenerator(
                featurewise_center=self.transforms['featurewise_center'],
                samplewise_center=self.transforms['samplewise_center'],
                featurewise_std_normalization=self.transforms['featurewise_std_normalization'],
                samplewise_std_normalization=self.transforms['samplewise_std_normalization'],
                zca_whitening=self.transforms['zca_whitening'],
                zca_epsilon=self.transforms['zca_epsilon'],
                rotation_range=self.transforms['rotation_range'],
                width_shift_range=self.transforms['width_shift_range'],
                height_shift_range=self.transforms['height_shift_range'],
                brightness_range=self.transforms['brightness_range'],
                shear_range=self.transforms['shear_range'],
                zoom_range=self.transforms['zoom_range'],
                channel_shift_range=self.transforms['channel_shift_range'],
                fill_mode=self.transforms['fill_mode'],
                cval=self.transforms['cval'],
                horizontal_flip=self.transforms['horizontal_flip'],
                vertical_flip=self.transforms['vertical_flip'],
                rescale=self.transforms['rescale'],
                preprocessing_function=self.transforms['preprocessing_function'],
                data_format=self.transforms['data_format'],
                validation_split=self.split_config['validation_split'],
                interpolation_order=self.transforms['interpolation_order'],
                dtype=self.transforms['dtype']
            )
        
        if (self.label_dir is not None) and (self.label_cols is not None):
            df = pd_read_file(self.label_dir, drop_duplicates=True)
        
            image_paths_col_name = self.label_cols[0]
            classes_col_name = self.label_cols[1]
            
            # drop nans only for target columns
            df.dropna(subset=[image_paths_col_name, classes_col_name], how='any', inplace=True)

            self.generator = self.data_gen.flow_from_dataframe(
                    df,
                    directory=self.img_dir,
                    x_col=image_paths_col_name,
                    y_col=classes_col_name,
                    classes=self.classes,
                    class_mode=self.class_mode,
                    target_size=self.imgs_target_config['target_size'],
                    interpolation=self.imgs_target_config['interpolation'],
                    color_mode=self.imgs_target_config['color_mode'],
                    save_to_dir=self.save_config['save_to_dir'],
                    save_format=self.save_config['save_format'],
                    validate_filenames=self.validate_filenames,
                    shuffle=self.shuffle_config['shuffle'],
                    seed=self.shuffle_config['seed'],
                    batch_size=self.batch_size
                )
        else:
            self.generator = self.data_gen.flow_from_directory(
                    directory=self.img_dir,
                    target_size=self.imgs_target_config['target_size'],
                    color_mode=self.imgs_target_config['color_mode'],
                    classes=self.classes,
                    class_mode=self.class_mode,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle_config['shuffle'],
                    seed=self.shuffle_config['seed'],
                    save_to_dir=self.save_config['save_to_dir'],
                    save_format=self.save_config['save_format'],
                    interpolation=self.imgs_target_config['interpolation'],
                )
    
    def __len__(self):
        """
        Calculates steps per epoch
        """
        return int(math.ceil((1. * self.generator.n) / self.batch_size))
    
    def get_img_gen(self):
        return self.generator


def test_img_data_gen_1():
    """
    Documenting how to use DataGenerator_img_disk when loading images from disk
    given that iamges sub directories names are the label names
    """
    
    # define your generator and required config
    generator = DataGenerator_img_disk(img_dir='images_parent directory',
                                    batch_size=8,
                                    class_mode='categorical',                       # categorical, binary, ...etc
                                    classes=['class 1', 'class 2'],                 # optional argument
                                    imgs_target_config={'target_size': (256, 256)}, 
                                    save_config={'save_to_dir': 'E:/augs', 'save_format': 'png'},   # saving augmentations config
                                    transforms={'rotation_range': 0.5,                              # transforms/augmentations config
                                                'horizontal_flip':True,
                                                'zoom_range':0.6,
                                                'brightness_range':[0.1, 0.5]})
    
    # get generator
    train_gen = generator.get_img_gen()

    # get generator length
    steps_per_epoch = generator.len()

    # use the generator in the fit function
    # model.fit(train_gen,
    #         epochs=3,
    #         verbose=1,
    #         steps_per_epoch=steps_per_epoch)

def test_img_data_gen_2():
    """
    Documenting how to use DataGenerator_img_disk when loading images from disk
    given that images paths and labels are in a csv file
    """
    
    # define your generator and required config
    generator = DataGenerator_img_disk(img_dir='images_parent directory',
                                    label_dir='CSV path', 
                                    label_cols=['x_col_name', 'y_col_name'],        # x_col (image paths) name, y_col (labels) columns name
                                    batch_size=8,
                                    class_mode='categorical',                       # categorical, binary, ...etc
                                    classes=['class 1', 'class 2'],                 # optional argument (can be infered from csv file)
                                    imgs_target_config={'target_size': (256, 256)}, 
                                    save_config={'save_to_dir': 'E:/augs', 'save_format': 'png'},   # saving augmentations config
                                    transforms={'rotation_range': 0.5,                              # transforms/augmentations config
                                                'horizontal_flip':True,
                                                'zoom_range':0.6,
                                                'brightness_range':[0.1, 0.5]})
    
    # get generator
    train_gen = generator.get_img_gen()

    # get generator length
    steps_per_epoch = generator.len()

    # use the generator in the fit function
    # model.fit(train_gen,
    #         epochs=3,
    #         verbose=1,
    #         steps_per_epoch=steps_per_epoch)




###############################################################################
from albumentations.core.transforms_interface import ImageOnlyTransform
class Transform_sprinkle(ImageOnlyTransform):
    def __init__(self, num_holes=30, side_length=5, always_apply=False, p=1.0):
        from tf_sprinkles import Sprinkles
        super(Transform_sprinkle, self).__init__(always_apply, p)
        self.sprinkles = Sprinkles(num_holes=num_holes, side_length=side_length)
    
    def apply(self, image, **params):
        if isinstance(image, PIL.Image.Image):   image = tf.constant(np.array(image), dtype=tf.float32)            
        elif isinstance(image, np.ndarray):      image = tf.constant(image, dtype=tf.float32)
        return self.sprinkles(image).numpy()


       
###############################################################################       
class DataGenerator_img_disk2(tf.keras.utils.Sequence):
    """Custom Data Generator using keras Sequence

        Args:
            image_dir (Path(str)): String Path /*.png to image directory
            label_path (DataFrame): Dataset for Generator
            class_dict (list): list of columns for categories
            split (str, optional): split as train, validation, or test. Defaults to 'train'.
            batch_size (int, optional): Batch size for the dataloader. Defaults to 8.
            transforms (str, optional): type of transform to perform on images. Defaults to None.
            shuffle (bool, optional): Shuffle the data. Defaults to True.
        """
        
    def __init__(self, image_dir, label_path, class_dict,
                 split='train', batch_size=8, transforms=None, shuffle=True):
        self.image_dir = image_dir
        # self.labels = np.loadtxt(label_dir, delimiter=' ', dtype=np.object)
        self.class_dict = class_dict
        self.image_ids, self.labels = self._load_data(label_path)
        self.num_classes = len(class_dict)
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle

    def _load_data(self, label_path):
        df = pd.read_csv(label_path, error_bad_lines=False, warn_bad_lines=False)
        keys = ['id'] + list(self.class_dict.keys())
        df = df[keys]

        # Get image ids
        df = df.dropna()
        image_ids = df['id'].tolist()
        df = df.drop('id', axis=1)
        labels = []
        for col in self.class_dict:
            categories = pd.get_dummies(df[col]).values
            labels.append(categories)
        return image_ids, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(12)
            indices = np.arange(len(self.image_ids))
            np.random.shuffle(indices)
            self.image_ids = self.image_ids[indices]
            self.labels = [label[indices] for label in self.labels]

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for image_id in batch_img_ids:
            # Load image
            image = np.array(Image.open(os.path.join(self.image_dir, '%d.jpg' % image_id)).convert('RGB'))
            batch_x.append(image)

        batch_y = []
        for y_head in self.labels:
            batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size, :])

        if self.transforms is not None:
            batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)
        return (idx, batch_x, *batch_y)


       
###############################################################################
#############  Utilities ######################################################
def _byte_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_tfrecord(x, tfrecord_out_path, max_records):
    extractor = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet',
        input_shape=(xdim, ydim, cdim),
        pooling='avg'
    )
    with tf.io.TFRecordWriter(tfrecord_out_path) as writer:
        id_cnt = 0
        for i, (_, images, *_) in enumerate(x):
            if i > max_records:
                break
            batch_embedding = extractor(images, training=False).numpy().tolist()
            for embedding in batch_embedding:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'id': _byte_feature(str(id_cnt).encode('utf-8')),
                    'embedding': _float_feature(embedding),
                }))
                writer.write(example.SerializeToString())
                id_cnt += 1
    return tfrecord_out_path


   
   
   
   
   
   
   
   
   
   
# class CustomDataGenerator(Sequence):
    
#     """Custom DataGenerator using keras Sequence

#     Args:
#         x (np array): The input samples from the dataset
#         y (np arrays): The label column from the dataset
#         batch_size (int, optional): batch size for the samples. Defaults to 32.
#         augmentations (str, optional): perform augmentations to the input samples. Defaults to None.
#     """
    
#     def __init__(self, x, y, batch_size=32, augmentations=None):
#         self.x          = x
#         self.y          = y
#         self.batch_size = batch_size
#         self.augment    = augmentations

#     def __len__(self):
#         return int(np.ceil(len(self.x) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = []
#         for y_head in self.y:
#             batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size])
        
#         if self.augment is not None:
#             batch_x = np.stack([self.augment(image=x)['image'] for x in batch_x], axis=0)
#         return (batch_x, *batch_y)




# class CustomDataGenerator_img(Sequence):
    
#     """Custom DataGenerator using Keras Sequence for images

#         Args:
#             img_dir (Path(str)): String path to images directory
#             label_dir (DataFrame): Dataset for Generator
#             label_cols (list): list of classes
#             split (str, optional): split for train or test. Defaults to 'train'.
#             batch_size (int, optional): batch_size for each batch. Defaults to 8.
#             transforms (str, optional):  type of transformations to perform on images. Defaults to None.
#     """
#     # """
#     #    df_label format :
#     #        id, uri, cat1, cat2, cat3, cat1_onehot, cat1_onehot, ....
#     # """
        
#     def __init__(self, img_dir, label_dir, label_cols,
#                  split='train', batch_size=8, transforms=None):
#         self.image_dir = img_dir
#         self.label_cols = label_cols
#         self.batch_size = batch_size
#         self.transforms = transforms

#         dflabels = pd.read_csv(label_dir)
#         self.labels = data_add_onehot(dflabels, img_dir, label_cols)

#     def on_epoch_end(self):
#         np.random.seed(12)
#         np.random.shuffle(self.labels)

#     def __len__(self):
#         return int(np.ceil(len(self.labels) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         # Create batch targets
#         df_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

#         batch_x = []
#         batch_y = []  # list of heads

#         for ii, x in df_batch.iterrows():
#             img = np.array(Image.open(x['uri']).convert('RGB'))
#             batch_x.append(img)

#         for ci in self.label_cols:
#             v = [x.split(",") for x in df_batch[ci + "_onehot"]]
#             v = np.array([[int(t) for t in vlist] for vlist in v])
#             batch_y.append(v)

#         if self.transforms is not None:
#             batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)

#         return (batch_x, *batch_y)
