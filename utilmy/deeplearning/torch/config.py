    
from box import Box

BATCH_SIZE = None

model_info = {'dataonly': {'rule': 0.0},
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

arg = Box({
    "dataurl":  "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv",
    "datapath": './cardio_train.csv',

    ##### Rules
    "rules": {},

    #####
    "train_ratio": 0.7,
    "validation_ratio": 0.1,
    "test_ratio": 0.2,

    ##### Training
    "seed": 42,
    "device": 'cpu',  ### 'cuda:0',
    "batch_size": 32,
    "epochs": 1,
    "early_stopping_thld": 10,
    "valid_freq": 1,
    'saved_filename' :'./model.pt',

})



arg.model_info = model_info
arg.merge = 'cat'
arg.input_dim = 20   ### 20
arg.output_dim = 1
# log(arg)



#### Rule Interface setup   ############################
arg.rules = {
        "rule_threshold":  129.5,
        "src_ok_ratio":      0.3,
        "src_unok_ratio":    0.7,
        "target_rule_ratio": 0.7,
        "rule_ind": 2,    ### Index of the colum Used for rule:  df.iloc[:, rule_ind ]
}

# arg.rules.loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
# arg.rules.loss_rule_calc = loss_rule_calc_cardio


# arg.rules.loss_rule_calc = rule_loss_calc_cardio



ARG = Box({
    'DATASET': {},
    'model_info' : {},
    'training_config' : {},
})

### ARG.DATASET

ARG.DATASET.PATH =  './cardio_train.csv'
ARG.DATASET.URL = 'https://github.com/caravanuden/cardio/raw/master/cardio_train.csv'


### ARG.MODEL_INFO
ARG.MODEL_INFO.TYPE = 'dataonly' 
ARG.MODEL_INFO.ALPHA = 0.5 # auto set to 1.0 if TYPE = 'dataonly'

### ARG.MODEL_A

ARG.MODEL_INFO.MODEL_A = Box()   #MODEL_RULE
ARG.MODEL_INFO.MODEL_A.NAME = ''
ARG.MODEL_INFO.MODEL_A.ARCHITECT = [
    16,
    100,
    16
]

### ARG.MODEL_B

ARG.MODEL_INFO.MODEL_B = Box()   #MODEL_RULE
ARG.MODEL_INFO.MODEL_B.NAME = ''
ARG.MODEL_INFO.MODEL_B.ARCHITECT = [
    16,
    100,
    16
]


ARG.MODEL_INFO.MODEL_MERGE = Box()
ARG.MODEL_INFO.MODEL_MERGE.NAME = ''
ARG.MODEL_INFO.MODEL_MERGE.ARCHITECT = {
    'encoder': [ARG.MODEL_INFO.MODEL_B,ARG.MODEL_INFO.MODEL_A],
    'merge' : 'add',
    'decoder': [
    32,  
    100,
    1
]
}