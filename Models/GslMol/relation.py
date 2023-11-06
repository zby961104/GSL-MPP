from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import time
import os
import math


INF = 1e20
VERY_SMALL_NUMBER = 1e-12


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

    def forward(self, x, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, adj)

        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input1 = torch.matmul(h, self.a1)
        a_input2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(a_input1 + a_input2.transpose(-1, -2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='attention', device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])

            self.leakyrelu = nn.LeakyReLU(0.2)

            print('[ GAT_Attention GraphLearner]')

        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)

        elif metric_type == 'cosine':
            pass

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF

        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF

        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0

        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)
        return attention

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(
            -1, knn_ind, knn_val).to(self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists


class FpSimNet(nn.Module):
    def __init__(self, opt, smiles_list):
        super(FpSimNet, self).__init__()
        self.opt = opt
        self.device = torch.device(
            f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if torch.cuda.is_available() else 'cpu')

        self.graph_skip_conn = opt.args['graph_skip_conn']
        # build or load init adj
        if self.graph_skip_conn not in (0, None):
            if opt.get_args('adjust init adj epsilon', False):
                self.init_adj_file = self.opt.args['ExpDir'] + 'init_adj.pt'
            else:
                self.init_adj_file = self.opt.args['TrialPath'] + 'init_adj.pt'

            if os.path.exists(self.init_adj_file):
                print('load init adj...')
                self.init_adj = torch.load(self.init_adj_file)
            else:
                self.init_adj = self.build_init_adj(
                    smiles_list, opt.args['radius'], opt.args['nBits'])

        if opt.args['map_layer'] == 0:
            self.inp_dim = opt.args['FPSize']
        else:
            self.inp_dim = opt.args['map_dim']

        self.graph_learn = opt.args['graph_learn']
        self.num_layers = opt.args['rel_layer']
        self.graph_metric_type = opt.args['rel_metric']
        self.rel_gnn = opt.get_args('rel_gnn', 'gcn')
        self.hidden_dim = opt.args['rel_hidden_dim']
        self.res_alpha = opt.args['rel_res']

        self.pre_dropout = opt.args['rel_dropout2']

        self.graph_include_self = opt.args['graph_include_self']
        self.update_adj_ratio = opt.args['update_adj_ratio']
        self.scalable_run = opt.get_args('scalable_run', False)

        if self.pre_dropout > 0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)

        if self.graph_learn:
            graph_learn_fun = AnchorGraphLearner if self.scalable_run else GraphLearner
            self.graph_learner1 = graph_learn_fun(input_size=self.inp_dim, hidden_size=self.hidden_dim,
                                              topk=opt.args['rel_k'], epsilon=opt.args['rel_epsilon'],
                                              num_pers=opt.args['rel_num_pers'], metric_type=opt.args['rel_metric'],
                                              device=self.device)

            self.graph_learner2 = graph_learn_fun(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                                              topk=opt.args['rel_k'], epsilon=opt.args['rel_epsilon'],
                                              num_pers=opt.args['rel_num_pers'], metric_type=opt.args['rel_metric'],
                                              device=self.device)
            print('[ Graph Learner ]')
        else:
            self.graph_learner1 = None
            self.graph_learner2 = None

        if self.rel_gnn == 'gcn':
            gcn_module = AnchorGCN if self.scalable_run else GCN
            self.encoder = gcn_module(nfeat=self.inp_dim, nhid=self.hidden_dim, graph_hops=opt.args['rel_gnn_layer'],
                               dropout=opt.args['rel_dropout'], batch_norm=opt.args['rel_batch_norm'])
        elif self.rel_gnn == 'gat':
            self.encoder = GAT(nfeat=self.inp_dim, nhid=self.hidden_dim, dropout=opt.args['rel_dropout'],
                               nclass=opt.args['ClassNum'],
                               nheads=opt.get_args('gat_nhead', 1), alpha=opt.get_args('gat_alpha', 0.2))
        # elif self.rel_gnn == 'graphsage':
        #     self.encoder = GraphSAGE(in_feats=self.inp_dim, n_hidden=self.hidden_dim,
        #                              n_classes=opt.args['ClassNum'], n_layers=opt.args['rel_gnn_layer'],
        #                              activation=F.relu, dropout=opt.args['rel_dropout'],
        #                              aggregator_type=opt.get_args('graphsage_agg_type', 'gcn'))
        else:
            raise RuntimeError('Unknown graph_module')

        self.fc1 = nn.Linear(self.hidden_dim, self.inp_dim)

        if self.pre_dropout > 0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)

        self.fc2 = nn.Linear(self.inp_dim, opt.args['ClassNum'])

        assert 0 <= self.res_alpha <= 1

    def build_init_adj(self, smiles_list, radius=2, nBits=1024):
        print('start to build init adj...')
        start = time.time()
        n = len(smiles_list)
        init_adj = torch.zeros((n, n), dtype=torch.float).to(self.device)
        fp_list = []
        for smiles in smiles_list:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smiles), radius, nBits=nBits)
            fp_list.append(fp)

        for i in range(n):
            for j in range(n):
                similarity = DataStructs.FingerprintSimilarity(fp_list[i], fp_list[j])
                if similarity < self.opt.args['init_adj_epsilon']:
                    similarity = 0.0
                init_adj[i][j] = similarity

        if self.opt.get_args('init_adj_norm', False):
            init_adj = init_adj / torch.clamp(torch.sum(init_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

        end = time.time()

        torch.save(init_adj, self.init_adj_file)

        print('init adj has been built, cost time=', end-start)
        return init_adj

    def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, graph_include_self=False, anchor_features=None):
        if self.graph_learn:
            if self.scalable_run:
                node_anchor_adj = graph_learner(node_features, anchor_features)
                return node_anchor_adj
            else:
                raw_adj = graph_learner(node_features)

                if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                    assert raw_adj.min().item() >= 0
                    adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

                elif self.graph_metric_type == 'cosine':
                    adj = (raw_adj > 0).float()
                    adj = self.normalize_adj(adj)

                else:
                    adj = torch.softmax(raw_adj, dim=-1)

                if graph_skip_conn in (0, None):
                    if graph_include_self:
                        adj = adj + torch.eye(adj.size(0)).to(self.device)
                else:
                    adj = graph_skip_conn * self.init_adj + (1 - graph_skip_conn) * adj

                return raw_adj, adj

        else:
            raw_adj = None
            adj = self.init_adj

            return raw_adj, adj

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def forward(self, features, return_adj=False):

        if self.pre_dropout > 0:
            features = self.predrop1(features)

        init_node_vec = features

        cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner1, init_node_vec, self.graph_skip_conn,
                                                self.graph_include_self)
        node_vec = self.encoder(init_node_vec, cur_adj)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if self.graph_learn:
            for i in range(self.num_layers - 1):

                cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2, node_vec, self.graph_skip_conn,
                                                        self.graph_include_self)

                cur_adj = self.update_adj_ratio * cur_adj + (1 - self.update_adj_ratio) * first_adj

                node_vec = self.encoder(init_node_vec, cur_adj)

        node_vec = self.fc1(node_vec)
        if self.pre_dropout > 0:
            node_vec = self.predrop2(node_vec)

        node_vec = self.res_alpha * features + node_vec

        pred = self.fc2(node_vec)

        if return_adj:
            if self.opt.get_args('return_raw_adj', True):
                return pred, cur_raw_adj, init_node_vec
            else:
                return pred, cur_adj, init_node_vec
        else:
            return pred

    def scalable_run(self, features, return_adj=False):
        if self.pre_dropout > 0:
            features = self.predrop1(features)

        init_node_vec = features

        init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, self.opt.get_args('num_anchors',
                                                                                             int(0.2 * init_node_vec.size(
                                                                                                 0))))
        cur_node_anchor_adj = self.learn_graph(self.graph_learner1, init_node_vec, anchor_features=init_anchor_vec)

        cur_anchor_adj = compute_anchor_adj(cur_node_anchor_adj)

        node_vec = self.encoder(init_node_vec, cur_adj)

        first_raw_adj, first_adj = cur_raw_adj, cur_adj

        if self.graph_learn:
            for i in range(self.num_layers - 1):

                cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner2, node_vec, self.graph_skip_conn,
                                                        self.graph_include_self)

                cur_adj = self.update_adj_ratio * cur_adj + (1 - self.update_adj_ratio) * first_adj

                node_vec = self.encoder(init_node_vec, cur_adj)

        node_vec = self.fc1(node_vec)
        if self.pre_dropout > 0:
            node_vec = self.predrop2(node_vec)

        node_vec = self.res_alpha * features + node_vec

        pred = self.fc2(node_vec)

        if return_adj:
            return pred, cur_raw_adj, init_node_vec
        else:
            return pred
