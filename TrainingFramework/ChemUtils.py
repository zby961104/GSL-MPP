import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


class ScaffoldGenerator(object):
    # Scaffold generator is used to generate scaffolds for scaffold splitting.
    def __init__(self, include_chirality = False):
        super(ScaffoldGenerator, self).__init__()
        self.include_chirality = include_chirality

    def get_scaffold(self, smiles):
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol = mol,
            includeChirality = self.include_chirality
        )


def GetMol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def GetSmiles(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles


def GetEdgeList(mol, bidirection=False, offset = 0):
    # input: mol - molecule
    #        bidirection - directional (distinguish (i,j) and (j,i), default False
    #        offset - offset for the atom index, default 0
    # output: edge_list - [num_edge, 2]
    bondlist = mol.GetBonds()
    edge_list = []
    bond_cnt = 0
    for bond in bondlist:
        bond_idx = bond.GetIdx()
        assert bond_cnt == bond_idx
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        edge_list.append([start_atom+offset, end_atom+offset])
        if bidirection:
            edge_list.append([end_atom+offset, start_atom+offset])
        bond_cnt += 1
    if len(edge_list) == 0:
        edge_list = np.empty((0,2), dtype=np.int64)
    return edge_list


def GetAtomFeatures(atom):
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    SymbolList = ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At']
    if symbol in SymbolList:
        loc = SymbolList.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    HybridizationList = [rdkit.Chem.rdchem.HybridizationType.SP,
                         rdkit.Chem.rdchem.HybridizationType.SP2,
                         rdkit.Chem.rdchem.HybridizationType.SP3,
                         rdkit.Chem.rdchem.HybridizationType.SP3D,
                         rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in HybridizationList:
        loc = HybridizationList.index(hyb)
        feature[loc+24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31+hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        feature[36] = 1
        try:
            chi = atom.GetProp('_CIPCode')
            ChiList = ['R','S']
            loc = ChiList.index(chi)
            feature[37+loc] = 1
            #print("Chirality resolving finished.")
        except:
            feature[37] = 0
            feature[38] = 0
    return feature


def GetBondFeatures(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    BondTypeList = [rdkit.Chem.rdchem.BondType.SINGLE,
                    rdkit.Chem.rdchem.BondType.DOUBLE,
                    rdkit.Chem.rdchem.BondType.TRIPLE,
                    rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in BondTypeList:
        loc = BondTypeList.index(type)
        feature[0+loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = conj

    # stereo
    stereo = bond.GetStereo()
    StereoList = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                  rdkit.Chem.rdchem.BondStereo.STEREOANY,
                  rdkit.Chem.rdchem.BondStereo.STEREOZ,
                  rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in StereoList:
        loc = StereoList.index(stereo)
        feature[6+loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature


def GetMolFeatureMat(mol):
    FeatureMat = []
    for atom in mol.GetAtoms():
        feature = GetAtomFeatures(atom)
        FeatureMat.append(feature.tolist())
    return FeatureMat


def GetNodeFeatureMat(mol):
    NodeFeatureMat = []
    for atom in mol.GetAtoms():
        node_feature = GetAtomFeatures(atom)
        NodeFeatureMat.append(node_feature.tolist())
    return NodeFeatureMat


def GetBondFeatureMat(mol, bidirection=False):
    FeatureMat = []
    for bond in mol.GetBonds():
        feature = GetBondFeatures(bond)
        FeatureMat.append(feature.tolist())
        if bidirection:
            FeatureMat.append(feature.tolist())

    # mol has no bonds
    if len(FeatureMat) == 0:
        FeatureMat = np.empty((0, 10), dtype = np.int64)
    else:
        FeatureMat = np.array(FeatureMat, dtype = np.int64)
    return FeatureMat


##########################################################
# Checkers to screen the dataset
##########################################################
class BasicChecker(object):
    def __init__(self):
        super(BasicChecker, self).__init__()

    def check(self, dataset):
        raise NotImplementedError(
            "Dataset Checker not implemented.")


class MolChecker(BasicChecker):
    def __init__(self):
        super(MolChecker, self).__init__()

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = GetMol(smiles)
            if mol:
                checked_dataset.append(item)
            else:
                discarded_dataset.append(item)
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules have passed check.")
        print(len(discarded_dataset), " molecules have been discarded.")
        print("Discarded molecules:")
        print(discarded_dataset)
        return checked_dataset


class AttentiveFPChecker(BasicChecker):
    # Rules proposed in the source code of Attentive FP
    # To screen the samples that not satisfy the rules
    # more rules can be added.
    def __init__(self, max_atom_num, max_degree):
        super(AttentiveFPChecker, self).__init__()
        self.max_atom_num = max_atom_num
        self.max_degree = max_degree
        self.mol_error_flag = 0

    def check(self, dataset):
        origin_dataset = dataset
        checked_dataset = []
        discarded_dataset = []
        for item in origin_dataset:
            smiles = item['SMILES']
            mol = GetMol(smiles)
            #check
            if mol:
                #self.check_single_bonds(mol)
                self.check_degree(mol)
                self.check_max_atom_num(mol)
                if self.mol_error_flag == 0:
                    checked_dataset.append(item)
                else:
                    discarded_dataset.append(item)
                    self.mol_error_flag = 0
            else:
                discarded_dataset.append(item)
                self.mol_error_flag = 0
        assert len(checked_dataset) + len(discarded_dataset) == len(origin_dataset)
        print("Total num of origin dataset: ", len(origin_dataset))
        print(len(checked_dataset), " molecules has passed check.")
        print(len(discarded_dataset), " molecules has been discarded.")
        print("Discarded molecules:")
        print(discarded_dataset)
        return checked_dataset

    def check_degree(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetDegree() > self.max_degree:
                self.mol_error_flag = 1
                break

    def check_max_atom_num(self, mol):
        if len(mol.GetAtoms()) > self.max_atom_num:
            self.mol_error_flag = 1

    def check_single_bonds(self, mol):
        # check whether there is at least one single bond in the molecule
        # this check is not used in FraGAT
        self.mol_error_flag = 1
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                if not bond.IsInRing():
                    self.mol_error_flag = 0
                    break
