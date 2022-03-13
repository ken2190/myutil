"""
hello,
are you here ?

yes.

ok good.

Can you see my commment below
yes.

reason is this: 

we have a base model ModelA (ie data_encoder)
and we want to plug a new model : model B (ie rule_encoder)


Your friend provide BlackBox model A (date encoder)
We create model B (rule encoder) and create the mergeEncCoder

---> Completelt separating --> allows to test separately models, debug them separattely.

ok ?
ok.
You want to build, train, evaluate model separattely?

if model A is black box, so i should separate model A as abstract class.

function create_model -> create torch.nn.Model is for define model architect.

you mean this architect is unknow?

and this architect should define in config file. 

can you here ?
https://docs.google.com/document/d/1l8x7pjynK5gyxawI7M_loQNZJm-7gfQ6FlSKsbvv6VI/edit?usp=sharing

easier for discusison
sorry


"""


We need to split INTO 3 completely SEPARATE Part 

COMPLETELY SEPARATE


    MergeEncoder(ARG,   data_encoer,  rule_encoder)



    ### data_encoder  #################################
    ARG.data_encoder = Box()   #MODEL_TASK
    ARG.data_encoder.NAME = ''
    ARG.data_encoder.ARCHITECT = [ 20, 100, 16 ]
    ARG.data_encoder.dataset = Box()
    ARG.data_encoder.dataset.dirin = "/"
    ARG.data_encoder.dataset.colsy =  'solvey'

    data_encoder = RuleEncoder_Create(ARG.data_encoder )



    ### rule_encoder  #################################
    ARG.rule_encoder = Box()   #MODEL_RULE
    ARG.rule_encoder.dataset = Box()
 
    ....
    ARG.rule_encoder.ARCHITECT = [9,100,16]
    rule_encoder = RuleEncoder_Create(ARG.rule_encoder )



    ### merge
    model = MergeEncoder_Create(ARG, rule_encoder, data_encoder)


    #### Run Model   ###################################################
    model.build()        
    model.training(prepro_dataset) 

    model.save_weight('ztmp/model_x9.pt') 
    model.load_weights('ztmp/model_x9.pt')

    ### Predict 
    inputs = torch.randn((1,9)).to(model.device)
    outputs = model.predict(inputs)
    print(outputs)





def test2_new():    
    """    
    """    
    from box import Box ; from copy import deepcopy
    ARG = Box({
        'DATASET': {},
        'MODEL_INFO' : {},
        'TRAINING_CONFIG' : {},
    })


    PARAMS = Box()

    if 'ARG':
        BATCH_SIZE = None

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


        ### ARG.MODEL_INFO
        ARG.MODEL_INFO.TYPE = 'dataonly' 
        PARAMS = MODEL_ZOO[ARG.MODEL_INFO.TYPE]


        #ARG.TRAINING_CONFIG
        ARG.TRAINING_CONFIG = Box()
        ARG.TRAINING_CONFIG.LR = PARAMS.get('lr',None)
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



    # load_DataFrame = RuleEncoder_Create.load_DataFrame   
    prepro_dataset = RuleEncoder_Create.prepro_dataset




    #### SEPARATE the models completetly, and create duplicate

    ### data_encoder  #################################
    ARG.data_encoder = Box()   #MODEL_TASK
    ARG.data_encoder.NAME = ''
    ARG.data_encoder.ARCHITECT = [ 20, 100, 16 ]
    ARG.data_encoder.dataset = Box()
    ARG.data_encoder.dataset.dirin = "/"
    ARG.data_encoder.dataset.colsy =  'solvey'

    data_encoder = RuleEncoder_Create(ARG.data_encoder )





    ### rule_encoder  #################################
    ARG.rule_encoder = Box()   #MODEL_RULE

    ARG.rule_encoder.dataset = Box()
    ARG.rule_encoder.dataset.dirin = "/"
    ARG.rule_encoder.dataset.colsy =  'solvey'


    ARG.rule_encoder.RULE = PARAMS.get('rule',None)
    ARG.rule_encoder.SCALE = PARAMS.get('scale',1.0)
    ARG.rule_encoder.PERT = PARAMS.get('pert',0.1)
    ARG.rule_encoder.BETA = PARAMS.get('beta',[1.0])
    beta_param = ARG.rule_encoder.BETA

    from torch.distributions.beta import Beta
    if   len(beta_param) == 1:  ARG.rule_encoder.ALPHA_DIST = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:  ARG.rule_encoder.ALPHA_DIST = Beta(float(beta_param[0]), float(beta_param[1]))



    ARG.rule_encoder.NAME = ''
    ARG.rule_encoder.RULE_IND = 2
    ARG.rule_encoder.RULE_THRESHOLD = 129.5
    ARG.rule_encoder.SRC_OK_RATIO = 0.3
    ARG.rule_encoder.SRC_UNOK_RATIO = 0.7
    ARG.rule_encoder.TARGET_RULE_RATIO = 0.7
    ARG.rule_encoder.ARCHITECT = [9,100,16]
    rule_encoder = RuleEncoder_Create(ARG.rule_encoder )




    ### merge_encoder  #################################
    ARG.merge_encoder = Box()
    ARG.merge_encoder.NAME = ''
    ARG.merge_encoder.SKIP = False
    ARG.merge_encoder.MERGE = 'cat'
    ARG.merge_encoder.ARCHITECT = { 'decoder': [ 32, 100, 1 ] }

    ARG.merge_encoder.dataset = Box()
    ARG.merge_encoder.dataset.dirin = "/"
    ARG.merge_encoder.dataset.colsy =  'solvey'


    model = MergeEncoder_Create(ARG, rule_encoder, data_encoder)


    #### Run Model   ###################################################
    model.build()        
    model.training(prepro_dataset) 

    model.save_weight('ztmp/model_x9.pt') 
    model.load_weights('ztmp/model_x9.pt')
    inputs = torch.randn((1,9)).to(model.device)
    outputs = model.predict(inputs)
    print(outputs)

