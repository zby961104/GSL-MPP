import torch
import torch.nn as nn

from .BasicGNNs import PyGGIN
from .relation import MLP, FpSimNet

VERY_SMALL_NUMBER = 1e-12


class MolRelationNet(nn.Module):
    def __init__(self, opt, smiles_list=None):
        super(MolRelationNet, self).__init__()
        self.rel_layer = opt.args['rel_layer']
        self.rel_metric = opt.args['rel_metric']

        self.mol_encoder = PyGGIN(opt, FeatureExtractor=True)

        self.task_num = opt.args['TaskNum']

        # if not opt.args['input_model_file'] == '':
        #     self.mol_encoder.from_pretrained(opt.args['input_model_file'])

        self.encode_projection = MLP(inp_dim=opt.args['FPSize'], hidden_dim=opt.args['map_dim'],
                                     num_layers=opt.args['map_layer'], dropout=opt.args['map_dropout'])
        if opt.args['map_layer'] > 0:
            inp_dim = opt.args['map_dim']
        else:
            inp_dim = opt.args['FPSize']

        if self.task_num == 1:
            self.relation_net = FpSimNet(opt, smiles_list)
        else:
            self.relation_nets = nn.ModuleList()
            for i in range(self.task_num):
                self.relation_nets.append(FpSimNet(opt, smiles_list))

    def to_one_hot(self, class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(0)
        # reshape
        label_i = label.repeat(1, num_samples)
        label_j = label_i.transpose(0, 1)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(0)).to(edge.device)
            edge=edge*diag_mask

        if self.rel_metric == 'weighted_cosine':
            edge = edge / edge.sum(-1).unsqueeze(-1)

        return edge

    def label2edge_r(self, label, mask_diag=True):
        # get size
        print(label.size())
        num_samples = label.size(0)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, num_samples)
        label_j = label_i.transpose(0, 1)
        print(label_i.size())
        print(label_j.size())
        # compute edge
        diff = torch.abs(label_i - label_j)
        boundary = 0.3
        one = torch.ones_like(diff)
        zero = torch.zeros_like(diff)
        edge = torch.where(diff < boundary, one, zero).float().to(label.device)

        # expand
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(0)).to(edge.device)
            edge = edge * diag_mask

        if self.edge_type == 'weighted_cosine':
            edge = edge / torch.clamp(torch.sum(edge, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        elif self.edge_activation == 'softmax':
            edge = edge / torch.clamp(torch.sum(edge, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        return edge

    def forward(self, data):

        mol_emb = self.mol_encoder(data)

        emb_map = self.encode_projection(mol_emb)

        # if self.rel_layer > 0:
        #     pred, adj = self.adapt_relation(emb_map, return_adj=True)
        #     return pred, adj
        # else:
        #     pred = self.adapt_relation(emb_map, return_adj=False)
        #     return pred
        if self.task_num == 1:
            pred, pred_adj, init_node_vec = self.relation_net(emb_map, return_adj=True)
            return pred, pred_adj, init_node_vec
        else:
            all_task_pred = []
            all_task_adj = []
            for i in range(self.task_num):
                # print(f'task {i}...')
                relation_net = self.relation_nets[i]
                pred, pred_adj, init_node_vec = relation_net(emb_map, return_adj=True)
                all_task_pred.append(pred)
                all_task_adj.append(pred_adj)
            all_task_pred = torch.cat(all_task_pred, dim=-1)

            return all_task_pred, all_task_adj, None

