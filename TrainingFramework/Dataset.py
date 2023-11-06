from TrainingFramework.FileUtils import *
from TrainingFramework.Splitter import *
from TrainingFramework.Featurizer import *
from torch.utils import data
from torch_geometric.data import InMemoryDataset
import os


class PyGMolDataset(InMemoryDataset):
    def __init__(self, graphdataset, opt, mode):
        self.graph_dataset = graphdataset
        self.opt = opt
        self.dataset_path_root = self.opt.args['ExpDir'] + 'Dataset/'
        if not os.path.exists(self.dataset_path_root):
            os.mkdir(self.dataset_path_root)
        self.mode = mode
        if os.path.exists(self.dataset_path_root + 'processed/' + self.processed_file_names[0]):
            os.remove(self.dataset_path_root + 'processed/' + self.processed_file_names[0])
        super(PyGMolDataset, self).__init__(root = self.dataset_path_root)
        self.data, self.slices = t.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.opt.args['DataPath']]

    @property
    def processed_file_names(self):
        return [self.opt.args['ExpName'] + '_' + self.mode + '.pt']

    def download(self):
        pass

    def process(self):
        data_list = self.graph_dataset
        data, slices = self.collate(data_list)
        # print("Processed without saving complete.")
        print("Saving processed files...")
        t.save((data, slices), self.processed_paths[0])
        print('Saving complete!')


class MolDatasetCreator(object):
    # An object to create molecule datasets from a given dataset file path.
    # Using CreateDatasets function to generate 2 or 3 datasets, based on the SplitRate
    # Based on the MolDatasetCreator above, this version added the MSN creating part
    # including the network building and the mask creating according to the splitting.
    def __init__(self, opt):
        super(MolDatasetCreator, self).__init__()
        self.FileParserList = {
            'HIV': HIVFileParser(),
            'BBBP': BBBPFileParser(),
            'Tox21': Tox21FileParser(),
            'FreeSolv': FreeSolvFileParser(),
            'ESOL': ESOLFileParser(),
            'QM9': QM9FileParser(),
            'BACE': BACEFileParser(),
            'ClinTox': ClinToxFileParser(),
            'SIDER': SIDERFileParser(),
            'SHP2': SHP2FileParser(),
            'Toxcast': ToxcastFileParser(),
            'Lipop': LipopFileParser(),
            'CEP': CEPFileParser(),
            'Malaria': MalariaFileParser(),
            'MUV': MUVFileParser(),
            'QM7': QM7FileParser()
        }
        self.SplitterList = {
            'Random': RandomSplitter(),
            'MultitaskRandom': MultitaskRandomSplitter(),
            'Scaffold': ScaffoldSplitter(),
            'ScaffoldRandom': ScaffoldRandomSplitter()
        }
        self.opt = opt

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == '0':
                    neg_count += 1
                elif value == '1':
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            # print('pos:', pos_weight, 'neg:', neg_weight)
            weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self, dataset = None):
        if not dataset:
            # read the dataset, get raw_data
            file_path = self.opt.args['DataPath']
            print("Loading data file...")
            fileloader = FileLoader(file_path)
            raw_data = fileloader.load()

            # parse raw_data, get raw_dataset
            print("Parsing lines...")
            parser = self.FileParserList[self.opt.args['ExpName']]
            raw_dataset = parser.parse_file(raw_data)
            print("Dataset is parsed. Original size is ", len(raw_dataset))

            print(f"Checking dataset...")
            self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
            self.screened_dataset = self.checker.check(raw_dataset)

            for idx, data in enumerate(self.screened_dataset):
                data.update({'idx': idx})

            self.CheckScreenedDatasetIdx()
        else:
            self.screened_dataset = dataset

        screened_smiles_list = []
        for item in self.screened_dataset:
            smiles = item['SMILES']
            screened_smiles_list.append(smiles)

        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None

        if self.opt.args['Splitter']:
            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets, idxs = splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)
            idxs = np.arange(len(self.screened_dataset))


        pyg_feater = PyGGraphFeaturizer(self.opt)

        PyGGraphset = []
        for sample in self.screened_dataset:
            graph_sample = pyg_feater.featurize(sample)
            PyGGraphset.append(graph_sample)

        all_set = PyGMolDataset(PyGGraphset, self.opt, 'ALL')

        return all_set, idxs, weights, screened_smiles_list


    def CheckScreenedDatasetIdx(self):
        print("Check whether idx is correct: ")
        chosen_idx = int(random.random() * len(self.screened_dataset))
        print(chosen_idx)
        print(self.screened_dataset[chosen_idx])
        assert chosen_idx == self.screened_dataset[chosen_idx]['idx']




