    
from box import Box

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
