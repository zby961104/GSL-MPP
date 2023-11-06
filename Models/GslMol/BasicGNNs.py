import torch as t
import torch.nn as nn
from torch_geometric.nn import GCN, global_add_pool, global_mean_pool, global_max_pool, MLP, GIN, MessagePassing

class PyGGIN(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGIN, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['GINInputSize']
        self.hidden_channel = opt.args['GINHiddenSize']
        self.out_channel = opt.args['FPSize']
        # self.eps = opt.args['GINEps']
        self.train_eps = opt.args['GINTrainEps']
        self.num_layers = opt.args['GINLayers']
        # self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor

        # self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        # self.GIN = GIN(in_channels = self.in_channel,
        #                hidden_channels = self.hidden_channel,
        #                out_channels = self.out_channel,
        #                num_layers = self.num_layers,
        #                dropout = self.dropout,
        #                eps = self.eps)
        self.GIN = GIN(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout,
                       train_eps = self.train_eps)

        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            # self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)
            self.TaskLayer = nn.Linear(self.out_channel, self.MLPOutputSize)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool,
        }
        self.readout = self.ReadoutList[opt.args['GINReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)

        x = self.GIN(x, Input.edge_index)
        # adj = SparseTensor(row=Input.edge_index[0], col=Input.edge_index[1], value=Input.edge_attr)
        # x = self.GIN(x, adj.t())

        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x

