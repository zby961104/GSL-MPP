import torch as t
from TrainingFramework.ChemUtils import *
from torch_geometric.data import Data


class PyGGraphFeaturizer(object):
    # Featurizer used for PyGGNNs, to generate a PyG.Data object from a given SMILES item.
    def __init__(self, opt):
        super(PyGGraphFeaturizer, self).__init__()
        self.opt = opt

    def featurize(self, item):
        SMILES = item['SMILES']
        Value = item['Value']
        mol = GetMol(SMILES)
        EdgeList = t.Tensor(GetEdgeList(mol)).long()
        AtomFeatureMat = t.Tensor(GetMolFeatureMat(mol))
        BondFeatureMat = t.Tensor(GetBondFeatureMat(mol))
        Label = self.GetLabelFromValues(Value)
        data = Data(x=AtomFeatureMat, edge_index = EdgeList.t().contiguous(), y = Label.t(), edge_attr = BondFeatureMat)
        return data

    def GetLabelFromValues(self, values):
        label = []
        if self.opt.args['ClassNum'] == 1:
            if self.opt.args['TaskNum'] == 1:
                label.append(float(values))
            else:
                for v in values:
                    label.append(float(v))
            label = t.Tensor(label)
        else:
            for v in values:
                label.append(int(v))
            label = t.Tensor(label).long()
        label.unsqueeze_(-1)

        return label









