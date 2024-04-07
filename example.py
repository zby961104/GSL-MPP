from TrainingFramework.ProcessControllers import *

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 3,
    'SeedSearch': False,
}

BasicParamList = {
    'ExpName': 'BBBP',
    'MainMetric': 'AUC',
    'DataPath': './Datasets/DrugData/BBBP_SMILESValue.txt',
    'RootPath': '/remote-home/byzhao/Experiments/GSL-MPP/try/',
    'CUDA_VISIBLE_DEVICES': '2',
    'TaskNum': 1,
    'ClassNum': 2,
    'OutputSize': 2,
    'Feature': 'PyGGIN',
    'Model': 'GslMol',
    'log_loss': False,

    'OnlySpecific': False,
    'Weight': True,
    'PyG': True,

    'ValidRate': 4000,
    'PrintRate': 2,
    'UpdateRate': 1,
    'ValidBalance': False,
    'TestBalance': False,
    'SplitRate': [0.8, 0.1],
    'Splitter': 'Scaffold',
    'MaxEpoch': 300,
    'LowerThanMaxLimit': 50,
    'DecreasingLimit': 30,

    'Scheduler': 'PolynomialDecayLR',

    # Params for PolynomialDecayLR only
    'WarmupEpoch': 2,
    'LRMaxEpoch': 300,
    'EndLR':1e-9,
    'Power':1.0,
    # Params for StepLR only
    'LRStep': 30,
    'LRGamma': 0.1,
    #####################################

    'WeightIniter': None,            #Choice: Norm, XavierNorm

    # Params for NormWeightIniter only
    'InitMean' : 0,
    'InitStd' : 1,

    #####################################

    'FeatureCategory': 'BaseOH',
    'AtomFeatureSize': 39,
    'BondFeatureSize': 10,

    #####################################

    # Params for GslMol only
    'init_adj_epsilon': 0.2,
    'radius': 2,
    'nBits': 1024,

    'graph_learn': True,
    'rel_layer': 2,
    'rel_hidden_dim': 128,

    # GslMol: graph_learner
    'rel_metric': 'weighted_cosine',
    'rel_k': None,
    'rel_num_pers': 16,
    'rel_epsilon': 0.0,
    'rel_res': 0,
    'graph_skip_conn': 0.8,
    'graph_include_self': False,
    'update_adj_ratio': 0.1,

    # GslMol: gnn_encoder
    'rel_gnn': 'gcn',
    'rel_gnn_layer': 2,
    'rel_batch_norm': True,

    'rel_dropout': 0,
    'rel_dropout2': 0.2,
    'adj_loss': True,

    # Params for PyGGIN only

    'GINLayers': 3,
    'GINInputSize': 32,
    'GINHiddenSize': 64,
    # 'GINEps': 0.1,
    'GINReadout': 'Add',
    'GINTrainEps': True,

    # Params for map-encoder of GslMol
    'map_dim': 128,
    'map_layer': 2,
    'map_dropout': 0.1,



    # Training Params to be adujsted. If the param is not needed to be adjusted, set the value here.
    'SplitSeedBias': 0,
    'FPSize': 32,
    'lr': None,
    'DropRate': None,
    'WeightDecay': None,
    'BatchSize': 200,
    'DNNLayers': [],

}
AdjustableParamList = {
    'FPSize' : [128, 32, 64, 150, 200],
    'GINLayers': [2,5,3,4],
    'GINInputSize': [32,128,64,256],
    'GINHiddenSize': [32,64,128,256],
    'GINTrainEps':[True, False],
    'GINReadout': ['Max', 'Add', 'Mean'],
    'DropRate' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

    'map_dim': [32, 64, 128],
    'map_layer': [0, 2, 1,  3],
    'map_dropout': [0.1, 0.2, 0.3, 0],

    # 'init_adj_epsilon': [0.7, 0.2, 0.3, 0.0, 0.1, 0.5],
    'rel_layer': [4,3, 1, 2, 5],
    'rel_hidden_dim': [256, 128,32, 64],
    'rel_gnn_layer': [2, 3],
    'rel_epsilon': [0.0, 0.1, 0.2, 0.5],
    'rel_num_pers': [8, 1, 2, 4,16],
    'rel_res': [1.0, 0.2, 0, 0.1,  0.5],
    # 'rel_metric': ['weighted_cosine', 'attention', 'gat_attention', 'kernel', 'transformer', 'cosine'],
    'rel_dropout': [0.0, 0.1, 0.2, 0.4, 0.6],
    'rel_dropout2': [0.1, 0.0, 0.2, 0.4, 0.6],
    'graph_skip_conn': [0.8, 0.7, 0.9, 0.5, 0.3, 0.1],
    'update_adj_ratio': [0.1, 0.3, 0.6, 0.8, 1.0],

    'adj_loss_weight': [0.5, 0.3, 0.1, 0.7, 0.9],

    'lr': [2.5, 3.5, 3, 2, 4],
    'WeightDecay': [5, 4, 4.5, 3.5, 3],

    'TorchSeed': [42, 8, 1234, 3407, 114514, 147258, 0, 1, 2],

}
SpecificParamList = {
    'TorchSeed': [42],
    'SplitSeedBias': [0],
    'lr': [2.5],
    'WeightDecay': [5],

    'adj_loss': [True],
    'adj_loss_weight': [0.5],
    'adjust init adj epsilon': [False],
    'init_adj_epsilon': [0.2],
    'init_adj_norm': [True],
    'return_raw_adj': [False],

    'rel_layer': [4],

    'rel_metric': ['weighted_cosine'],
    'rel_num_pers': [8],

    'rel_hidden_dim': [256],
    'rel_gnn': ['gcn'],
    'rel_gnn_layer': [2],
    'rel_batch_norm': [True],

    'rel_epsilon': [0.0],
    'rel_res': [1.0],
    'rel_dropout': [0],
    'rel_dropout2': [0.1],
    'graph_skip_conn': [0.8],
    'update_adj_ratio': [0.1],

    'FPSize': [32],

    'GINLayers': [2],
    'GINInputSize': [32],
    'GINHiddenSize': [32],
    # 'GINEps': 0.1,
    'GINReadout': ['Max'],
    'GINTrainEps': [True],
    'DropRate': [0],

    # Params for map-encoder of GslMol
    'map_dim': [32],
    'map_layer': [0],
    'map_dropout': [0.1],
}

expcontroller = ExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()