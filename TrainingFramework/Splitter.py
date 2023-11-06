import random
import torch
from TrainingFramework.ChemUtils import *


class BasicSplitter(object):
    # A splitter module is used to split a dataset
    # with a entire dataset given, the splitter will return the index of the samples of different subsets,
    # or return the subsets directly.
    # return: (sets), (sets_index)
    def __init__(self):
        super(BasicSplitter, self).__init__()

    def split(self, dataset, opt):
        raise NotImplementedError(
            'Dataset splitter not implemented.'
        )


class RandomSplitter(BasicSplitter):
    # Module for randomly splitting dataset
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def CheckClass(self, dataset, tasknum):
        # To check whether both classes of samples appear in the dataset.
        c0cnt = np.zeros(tasknum)
        c1cnt = np.zeros(tasknum)
        for data in dataset:
            value = data['Value']
            assert tasknum == len(value)
            for task in range(tasknum):
                if (value[task] == '0') or (value[task] == 0):
                    c0cnt[task] += 1
                elif (value[task] == '1') or (value[task] == 1):
                    c1cnt[task] += 1
        if 0 in c0cnt:
            print("Invalid splitting.")
            return False
        elif 0 in c1cnt:
            print("Invalid splitting.")
            return False
        else:
            return True

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        np_dataset = np.array(dataset)
        index = np.arange(total_num)

        if len(rate) == 1:
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
            endflag = 0

            while not endflag:
                random.seed(validseed)
                random.shuffle(index)
                set1_idx = index[:train_num]
                set2_idx = index[train_num:]

                assert len(set1_idx) == train_num
                assert len(set2_idx) == valid_num

                set1 = np_dataset[set1_idx]
                set2 = np_dataset[set2_idx]
                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1
            return (set1, set2), (set1_idx, set2_idx)

        if len(rate) == 2:
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
            endflag = 0

            while not endflag:
                random.seed(testseed)
                random.shuffle(index)
                set3_idx = index[(train_num + valid_num):]
                set3 = np_dataset[set3_idx]

                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set3, opt.args['TaskNum'])
                    testseed += 1
                else:
                    endflag = 1

            set_idx_remain = index[:(train_num + valid_num)]
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(set_idx_remain)

                set1_idx = set_idx_remain[:train_num]
                set2_idx = set_idx_remain[train_num:]
                set1 = np_dataset[set1_idx]
                set2 = np_dataset[set2_idx]

                if opt.args['ClassNum'] == 2:
                    endflag = self.CheckClass(set2, opt.args['TaskNum'])
                    validseed += 1
                else:
                    endflag = 1

                assert len(set1) == train_num
                assert len(set2) == valid_num
                assert len(set3) == test_num

            return (set1, set2, set3), (set1_idx, set2_idx, set3_idx)


class MultitaskRandomSplitter(BasicSplitter):
    def __init__(self):
        super(MultitaskRandomSplitter, self).__init__()

    def BuildEntireDataset(self, dataset):
        EntireDataset = {'Index':[], 'Items': []}
        for i in range(len(dataset)):
            EntireDataset['Index'].append(i)
            EntireDataset['Items'].append(dataset[i])
        return EntireDataset

    def merge(self, EntireDataset, EntireSet, set2_index, set2_value, task, task_num):
        # entire_set: {'Index': [], 'Items': [{'SMILES':, 'Value':}]}
        # set2_index: list of index of the set to be merged.
        # set2_value: value of the set to be merged
        NewSet = EntireSet.copy()
        EntireSetIndex = EntireSet['Index']
        for j in range(len(set2_index)):
            item_ind = set2_index[j]
            item_value = set2_value[j]
            if item_ind in EntireSetIndex:
                loc = EntireSetIndex.index(item_ind)
                NewSet['Items'][loc]['Value'][task] = item_value
            else:
                NewSet['Index'].append(item_ind)
                SMILES = EntireDataset['Items'][item_ind]['SMILES']
                Value = []
                for k in range(task_num):
                    Value.append(-1)
                Value[task] = item_value
                NewSet['Items'].append({'SMILES': SMILES, 'Value':Value})
        return NewSet


    def OneTaskSplit(self, EntireDataset, task, rate, validseed=0, testseed=0):
        print('Splitting task', task)
        EntireNum = len(EntireDataset['Index'])
        TaskIndexPosSet = []
        TaskIndexNegSet = []
        for i in range(EntireNum):
            data = EntireDataset['Items'][i]
            if (data['Value'][task] == '0') or (data['Value'][task] == 0):
                TaskIndexNegSet.append(i)
            elif (data['Value'][task] == '1') or (data['Value'][task] == 1):
                TaskIndexPosSet.append(i)
        TaskPosNum = len(TaskIndexPosSet)
        TaskNegNum = len(TaskIndexNegSet)
        print("TaskPosNum:", TaskPosNum)
        print("TaskNegNum:", TaskNegNum)

        if len(rate) == 1:
            TaskPosTrainNum = int(TaskPosNum * rate[0])
            TaskPosValidNum = TaskPosNum - TaskPosTrainNum
            TaskNegTrainNum = int(TaskNegNum * rate[0])
            TaskNegValidNum = TaskNegNum - TaskNegTrainNum
            assert TaskPosValidNum > 0
            assert TaskPosTrainNum > 0
            assert TaskNegValidNum > 0
            assert TaskNegTrainNum > 0

            random.seed(validseed)
            random.shuffle(TaskIndexPosSet)
            random.shuffle(TaskIndexNegSet)
            TaskPosTrainSet = TaskIndexPosSet[:TaskPosTrainNum]
            TaskPosValidSet = TaskIndexPosSet[TaskPosTrainNum:]
            TaskNegTrainSet = TaskIndexNegSet[:TaskNegTrainNum]
            TaskNegValidSet = TaskIndexNegSet[TaskNegTrainNum:]

            TaskTrainSet = TaskPosTrainSet + TaskNegTrainSet
            TaskTrainValueSet = []
            for i in range(len(TaskPosTrainSet)):
                TaskTrainValueSet.append(1)
            for i in range(len(TaskNegTrainSet)):
                TaskTrainValueSet.append(0)
            TaskValidSet = TaskPosValidSet + TaskNegValidSet
            TaskValidValueSet = []
            for i in range(len(TaskPosValidSet)):
                TaskValidValueSet.append(1)
            for i in range(len(TaskNegValidSet)):
                TaskValidValueSet.append(0)

            assert len(TaskTrainSet) == TaskPosTrainNum + TaskNegTrainNum
            assert len(TaskValidSet) == TaskPosValidNum + TaskNegValidNum

            return (TaskTrainSet, TaskValidSet), (TaskTrainValueSet, TaskValidValueSet)

        elif len(rate) == 2:
            TaskPosTrainNum = int(TaskPosNum * rate[0])
            TaskPosValidNum = int(TaskPosNum * rate[1])
            TaskPosTestNum = TaskPosNum - TaskPosTrainNum - TaskPosValidNum
            TaskNegTrainNum = int(TaskNegNum * rate[0])
            TaskNegValidNum = int(TaskNegNum * rate[1])
            TaskNegTestNum = TaskNegNum - TaskNegTrainNum - TaskNegValidNum

            assert TaskPosTrainNum > 0
            assert TaskPosValidNum > 0
            assert TaskPosTestNum > 0
            assert TaskNegTrainNum > 0
            assert TaskNegValidNum > 0
            assert TaskNegTestNum > 0

            random.seed(testseed)
            random.shuffle(TaskIndexPosSet)
            random.shuffle(TaskIndexNegSet)
            TaskPosRemainSet = TaskIndexPosSet[:(TaskPosTrainNum + TaskPosValidNum)]
            TaskPosTestSet = TaskIndexPosSet[(TaskPosTrainNum + TaskPosValidNum):]
            TaskNegRemainSet = TaskIndexNegSet[:(TaskNegTrainNum + TaskNegValidNum)]
            TaskNegTestSet = TaskIndexNegSet[(TaskNegTrainNum + TaskNegValidNum):]

            random.seed(validseed)
            random.shuffle(TaskPosRemainSet)
            random.shuffle(TaskNegRemainSet)
            TaskPosTrainSet = TaskPosRemainSet[:TaskPosTrainNum]
            TaskPosValidSet = TaskPosRemainSet[TaskPosTrainNum:]
            TaskNegTrainSet = TaskNegRemainSet[:TaskNegTrainNum]
            TaskNegValidSet = TaskNegRemainSet[TaskNegTrainNum:]

            TaskTrainSet = TaskPosTrainSet + TaskNegTrainSet
            TaskTrainValueSet = []
            for i in range(len(TaskPosTrainSet)):
                TaskTrainValueSet.append(1)
            for i in range(len(TaskNegTrainSet)):
                TaskTrainValueSet.append(0)
            TaskValidSet = TaskPosValidSet + TaskNegValidSet
            TaskValidValueSet = []
            for i in range(len(TaskPosValidSet)):
                TaskValidValueSet.append(1)
            for i in range(len(TaskNegValidSet)):
                TaskValidValueSet.append(0)
            TaskTestSet = TaskPosTestSet + TaskNegTestSet
            TaskTestValueSet = []
            for i in range(len(TaskPosTestSet)):
                TaskTestValueSet.append(1)
            for i in range(len(TaskNegTestSet)):
                TaskTestValueSet.append(0)

            assert len(TaskTrainSet) == TaskPosTrainNum + TaskNegTrainNum
            assert len(TaskValidSet) == TaskPosValidNum + TaskNegValidNum
            assert len(TaskTestSet) == TaskPosTestNum + TaskNegTestNum

            return (TaskTrainSet, TaskValidSet, TaskTestSet), (TaskTrainValueSet, TaskValidValueSet, TaskTestValueSet)

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        task_num = opt.args['TaskNum']
        EntireDataset = self.BuildEntireDataset(dataset)

        EntireTrainSet = {'Index':[], 'Items':[]}
        EntireValidSet = {'Index':[], 'Items':[]}
        EntireTestSet = {'Index':[], 'Items':[]}

        for i in range(task_num):
            sets, values = self.OneTaskSplit(EntireDataset, task = i, rate = rate, validseed = validseed, testseed = testseed)
            if len(rate) == 1:
                task_train_set, task_valid_set = sets
                task_train_value, task_valid_value = values
                EntireTrainSet = self.merge(EntireDataset, EntireTrainSet, task_train_set, task_train_value, i, task_num)
                EntireValidSet = self.merge(EntireDataset, EntireValidSet, task_valid_set, task_valid_value, i, task_num)

            elif len(rate) == 2:
                task_train_set, task_valid_set, task_test_set = sets
                task_train_value, task_valid_value, task_test_value = values
                EntireTrainSet = self.merge(EntireDataset, EntireTrainSet, task_train_set, task_train_value, i, task_num)
                EntireValidSet = self.merge(EntireDataset, EntireValidSet, task_valid_set, task_valid_value, i, task_num)
                EntireTestSet = self.merge(EntireDataset, EntireTestSet, task_test_set, task_test_value, i, task_num)

        if len(rate)==1:
            return (EntireTrainSet['Items'], EntireValidSet['Items']), (EntireTrainSet['Index'], EntireValidSet['Index'])
        if len(rate)==2:
            return (EntireTrainSet['Items'], EntireValidSet['Items'], EntireTestSet['Items']),\
                   (EntireTrainSet['Index'], EntireValidSet['Index'], EntireTestSet['Index'])
        # for each task:
        # First, extract data with valid values on task_i
        # Second, divide extracted data into positive and negative sets
        # Third, split positive set and negative set into train/valid/test sets by rates
        # merge pos and neg into train_task, valid_task, test_task sets.
        # Merge train_task with entire_task, the same as valid and test.

        # To merge pos and neg sets, because they are not overlapping, so it is easy to merge.
        # To merge task set and the entire set:
        # In the entire set, if an item is added into this set, its smiles and index will be stored, and the Values are set to be a vector with task_num length and all of the values are -1.
        # Then, for task_i, the Value[i] of this item will set to be either 1 or 0.
        # And for task_j, when adding an item, first


class ScaffoldRandomSplitter(BasicSplitter):
    def __init__(self):
        super(ScaffoldRandomSplitter, self).__init__()
        self.generator = ScaffoldGenerator()

    def generate_scaffold(self, smiles):
        scaffold = self.generator.get_scaffold(smiles)
        return scaffold

    def generate_all_scaffold(self, dataset):
        # input: the dataset of molecules
        # return: a dict of scaffolds appear in the dataset.
        #         for each scaffold, a list of idx which records the idx of samples that have this scaffold is recorded.
        # type of scaffolds: {'XXXXXXX' : [id1, id2, id3, ...]}
        scaffolds = {}
        for id, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)
        return scaffolds

    def multitask_generate_scaffold_keys(self, dataset):
        """
        Generate a scaffold dict and scaffold keys for each data.
            Parameters:
                dataset (Dataset): dataset to split
        """
        scaffold2id = {}
        keys = []
        for _, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffold2id:
                id = len(scaffold2id)
                scaffold2id[scaffold] = id
            else:
                id = scaffold2id[scaffold]
            keys.append(id)
        return keys

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        # globalseed = opt.args['MultiSplitSeed']
        total_num = len(dataset)

        # extract scaffolds of the dataset
        # multitask classification
        if (opt.args['TaskNum'] > 1) & (opt.args['ClassNum'] == 2):
            self.keys = self.multitask_generate_scaffold_keys(dataset)
        else:
            self.scaffolds = self.generate_all_scaffold(dataset)

        # for scaffold random selection, the scaffolds does not need to be sorted.

        # calculate the splitting thres
        if len(rate) == 1:
            assert rate[0] < 1
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
        elif len(rate) == 2:
            assert rate[0]+rate[1] < 1
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
        else:
            print("Wrong splitting rate")
            raise RuntimeError

        # For different task# and class#, the processing method is different.
        # While in the current version, only the method for 1 task 2 classes has been implemented.
        tasknum = opt.args['TaskNum']
        classnum = opt.args['ClassNum']  # for regression task, classnum is set to be 1

        if classnum == 1: # case: regression
            scaffold_keys = self.scaffolds.keys()  # sample scaffods from scafffold_keys
            if len(rate) == 1:  # only sample the validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0]))
                # trick: the ratio of the sampled scaffolds is equal to the ratio of the sampled molecules.
                # which means that the program hope to sample the scaffold_sets in an average size.
                # So that the scaffold_set sampled is not too small.
                validids, _ = self.RegressionSample(self.scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset), (trainids, validids)
            elif len(rate) == 2:  # sample testset then validset.
                sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
                testids, chosen_scaffolds = self.RegressionSample(self.scaffolds, sample_size, test_num, testseed)
                testset = self.id2data(dataset, testids)
                # remain_scaffolds = self.excludedscaffolds(scaffold_keys, chosen_scaffolds)
                remain_scaffolds = {x: self.scaffolds[x] for x in self.scaffolds.keys() if x not in chosen_scaffolds}
                sample_size = int(len(remain_scaffolds.keys()) * rate[1])
                validids, _ = self.RegressionSample(remain_scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids + testids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset, testset), (trainids, validids, testids)

        elif classnum == 2:   # case: binary classification
            if tasknum == 1:    # case: single task
                minorclass, minor_count, minor_ratio = self.BinarySingleTaskMinorClassCount(dataset)
                # else:    # case: multi-task
                #     minorclass, minor_count, minor_ratio = self.BinaryMultiTaskMinorClassCount(dataset)
                # minorclass is an string of '0' or '1'
                # minor_count is the number of entire minor class samples in the dataset.
                # minor_ratio is the ratio of the minor class samples in the dataset.
                # minor_count is not useful.
                scaffold_keys = self.scaffolds.keys()  # sample scaffolds from scaffold_keys
            #multitask
            else:
                pass
            if len(rate) == 1:  # only sample the validset.
                if tasknum == 1:
                    sample_size = int(len(scaffold_keys) * (1 - rate[0]))
                    # trick: the ratio of the sampled scaffolds is equal to the ratio of the sampled molecules.
                    # which means that the program hope to sample the scaffold_sets in an average size.
                    # So that the scaffold_set sampled is not too small.
                    validids, _ = self.BinaryClassSample(dataset, self.scaffolds, sample_size, valid_num, minor_ratio, minorclass, validseed)
                    trainids = self.excludedids(len(dataset), validids)
                #multitask
                else:
                    sample_size = valid_num
                    try_times = 20000
                    while try_times:
                        try:
                            torch.manual_seed(validseed)
                            trainids, validids = self.BinaryMultiTaskKeySplit(dataset, self.keys, sample_size)
                            if self.BinaryMultiTaskCheck(dataset, validids, tasknum):
                                break
                            else:
                                pass
                        except RuntimeError:
                            try_times = try_times - 1
                            validseed += 1
                            if try_times == 0:
                                raise RuntimeError
                        else:
                            pass
                validset = self.id2data(dataset, validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset), (trainids, validids)
            elif len(rate) == 2:  # sample testset then validset.
                if tasknum == 1:
                    sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
                    testids, chosen_scaffolds = self.BinaryClassSample(dataset, self.scaffolds, sample_size, test_num, minor_ratio,
                                                                       minorclass, testseed)
                    # remain_scaffolds = self.excludedscaffolds(scaffold_keys, chosen_scaffolds)
                    remain_scaffolds = {x: self.scaffolds[x] for x in self.scaffolds.keys() if x not in chosen_scaffolds}
                    sample_size = int(len(remain_scaffolds.keys()) * rate[1])
                    validids, _ = self.BinaryClassSample(dataset, remain_scaffolds, sample_size, valid_num, minor_ratio, minorclass,
                                                         validseed)
                    trainids = self.excludedids(len(dataset), validids + testids)
                #multitask
                else:
                    try_times = 20000
                    #check total dataset
                    self.BinaryMultiTaskCheck(dataset, list(range(total_num)), tasknum)
                    while try_times:
                        try:
                            torch.manual_seed(testseed)
                            ids = list(range(total_num))
                            sample_size = test_num
                            testids, remain_keys, remain_ids = self.BinaryMultiTaskKeySplit(self.keys, self.keys, ids, sample_size)
                            if self.BinaryMultiTaskCheck(dataset, testids, tasknum):
                                print(f"testseed:{testseed}")
                                break
                        except ValueError:
                            try_times = try_times -1
                            testseed += 1
                            if try_times == 0:
                                raise RuntimeError
                        else:
                            pass
                    while try_times:
                        try:
                            torch.manual_seed(validseed)
                            sample_size = valid_num
                            validids, _, trainids = self.BinaryMultiTaskKeySplit(self.keys, remain_keys, remain_ids, sample_size)
                            if self.BinaryMultiTaskCheck(dataset, validids, tasknum):
                                print(f"validseed:{validseed}")
                                print(f"split successfully!")
                                break
                            else:
                                pass
                        except ValueError:
                            try_times = try_times - 1
                            validseed += 1
                            if try_times == 0:
                                raise RuntimeError
                        else:
                            pass
                testset = self.id2data(dataset, testids)
                validset = self.id2data(dataset, validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset, testset), (trainids, validids, testids)

    def RegressionSample(self, scaffolds, sample_size, optimal_count, seed):
        # Inputs:
        # scaffolds: the scaffold dict
        # sample_size: the number of the scaffolds that are expected to select
        # optimal_count: the optimal number of molecules that are expected to select
        # seed: random seed.
        count = 0
        keys = list(scaffolds.keys())
        tried_times = 0
        error_rate = 0.1

        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)):
            tried_times += 1

            if tried_times % 5000 == 0:
                # too many tried times. Need to loosen the criterion
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample scaffold number.")
                sample_size = int(sample_size * 1.1)
                print(sample_size)
                assert sample_size < len(keys)

            seed += 1
            random.seed(seed)
            chosen_scaffolds = random.sample(keys, sample_size)
            count = sum([len(scaffolds[scaffold]) for scaffold in chosen_scaffolds])
            index = [index for scaffold in chosen_scaffolds for index in scaffolds[scaffold]]

        print("Sample num: ", count)
        print("Available Seed: ", seed)
        print("Tried times: ", tried_times)
        return index, chosen_scaffolds

    def BinaryClassSample(self, dataset, scaffolds, sample_size, optimal_count, minor_ratio, minor_class, seed):
        # minor_ratio: the ratio of the minor class samples in the dataset.
        # minor_class: to figure out which class is the minor class
        optimal_minor_count = minor_ratio * optimal_count
        count = 0
        minor_count = 0
        keys = list(scaffolds.keys())
        tried_times = 0
        error_rate = 0.1

        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)) \
                or (minor_count < optimal_minor_count * (1-error_rate)) or (minor_count > optimal_minor_count * (1+error_rate)):
            tried_times += 1
            # too many times have tried. Loosen the critierion.
            if tried_times % 5000 == 0:
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample scaffold number.")
                sample_size = int(sample_size * 1.1)
                print(sample_size)
                assert sample_size < len(keys)

            seed += 1
            random.seed(seed)
            chosen_scaffolds = random.sample(keys, sample_size)
            count = sum([len(scaffolds[scaffold]) for scaffold in chosen_scaffolds])
            index = [index for scaffold in chosen_scaffolds for index in scaffolds[scaffold]]
            minor_count = self.id2valuecount(dataset, index, minor_class) # count the number of samples of minor class

        print("Sample num: ", count)
        print("Minor class num: ", minor_count)
        print("Available Seed: ", seed)
        print("Tried times: ", tried_times)
        return index, chosen_scaffolds

    def id2valuecount(self, dataset, ids, count_value):
        count = 0
        for id in ids:
            data = dataset[id]
            value = data['Value']
            if value == count_value:
                count += 1
        return count

    def id2data(self, dataset, ids):
        new_dataset = []
        for id in ids:
            data = dataset[id]
            new_dataset.append(data)
        return new_dataset

    def BinarySingleTaskMinorClassCount(self, dataset):
        # Find out which class is the minor class
        class0_count = 0
        class1_count = 0
        for data in dataset:
            value = data['Value']
            if value == '0':
                class0_count += 1
            elif value == '1':
                class1_count += 1
            else:
                print("Value count error.")
                raise RuntimeError

        if class0_count > class1_count:
            minorclass = '1'
            minor_count = class1_count
            minor_ratio = class1_count / len(dataset)
        else:
            minorclass = '0'
            minor_count = class0_count
            minor_ratio = class0_count / len(dataset)

        return minorclass, minor_count, minor_ratio

    def BinaryMultiTaskMinorClassCount(self, dataset):
        # Find out which class is the minor class
        class0_count = 0
        class1_count = 0
        for data in dataset:
            value = data['Value']
            if value[0] == '0':
                class0_count += 1
            elif value[0] == '1':
                class1_count += 1
            else:
                print("Value count error.")
                raise RuntimeError

        if class0_count > class1_count:
            minorclass = '1'
            minor_count = class1_count
            minor_ratio = class1_count / len(dataset)
        else:
            minorclass = '0'
            minor_count = class0_count
            minor_ratio = class0_count / len(dataset)

        return minorclass, minor_count, minor_ratio

    def BinaryMultiTaskKeySplit(self, global_keys, keys, ids, sample_size):
        global_keys = torch.as_tensor(global_keys)
        keys = torch.as_tensor(keys)
        ids = torch.as_tensor(ids)
        key_set, keys = torch.unique(keys, return_inverse=True)
        perm = torch.randperm(len(key_set))
        keys = perm[keys]
        indexes = keys.argsort().tolist()
        offset = self.round_to_boundary(sample_size, keys, indexes)
        if offset == 0 or offset == len(keys):
            raise ValueError
        assert offset < len(indexes)
        set_ids = ids[indexes[0: offset]].tolist()
        ids = ids.tolist()
        for id in set_ids:
            ids.remove(id)
            global_keys[id] = -1
        global_keys = global_keys[global_keys != -torch.ones(len(global_keys))]
        remain_keys = global_keys.tolist()
        remain_ids = ids
        return set_ids, remain_keys, remain_ids

    def round_to_boundary(self, i, keys, indexes):
        #
        for j in range(min(i, len(keys) - i)):
            if keys[indexes[i - j]] != keys[indexes[i - j - 1]]:
                return i - j
            if keys[indexes[i + j]] != keys[indexes[i + j - 1]]:
                return i + j
        if i < len(keys) - i:
            return 0
        else:
            return len(keys)

    def BinaryMultiTaskCheck(self, dataset, ids, task_num):
        positive_count = np.zeros(task_num)
        negative_count = np.zeros(task_num)
        for id in ids:
            data = dataset[id]
            value = data['Value']
            assert task_num == len(value)
            for task in range(task_num):
                if value[task] == '-1':
                    pass
                elif value[task] == '0':
                    negative_count[task] += 1
                elif value[task] == '1':
                    positive_count[task] += 1
        if 0 in positive_count:
            print("No label : one")
            raise ValueError
        elif 0 in negative_count:
            print("No label: zero")
            raise ValueError
        else:
            print("check successfully.")
            return True

    def excludedids(self, total_num, ids):
        # return the remain idx after chosen sample.
        excludedids = []
        ids.sort()
        j = 0
        for i in range(total_num):
            if j < len(ids):
                if i != ids[j]:
                    excludedids.append(i)
                else:
                    j += 1
            else:
                excludedids.append(i)
        assert len(excludedids) + len(ids) == total_num
        return excludedids


class ScaffoldSplitter(BasicSplitter):
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
        self.generator = ScaffoldGenerator()

    def generate_scaffold(self, smiles):
        scaffold = self.generator.get_scaffold(smiles)
        return scaffold

    def generate_all_scaffold(self, dataset):
        # input: the dataset of molecules
        # return: a dict of scaffolds appear in the dataset.
        #         for each scaffold, a list of idx which records the idx of samples that have this scaffold is recorded.
        # type of scaffolds: {'XXXXXXX' : [id1, id2, id3, ...]}
        scaffolds = {}
        for id, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)
        return scaffolds

    def id2data(self, dataset, ids):
        new_dataset = []
        for id in ids:
            data = dataset[id]
            new_dataset.append(data)
        return new_dataset

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        total_num = len(dataset)

        assert rate[0]+rate[1] < 1
        train_num = int(total_num * rate[0])
        valid_num = int(total_num * rate[1])
        test_num = total_num - train_num - valid_num

        scaffolds = self.generate_all_scaffold(dataset)

        index_sets = list(scaffolds.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > valid_num / 2 or len(index_set) > test_num / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(opt.args['split_seed'])
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets


        trainids = []
        validids = []
        testids = []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

        for index_set in index_sets:
            if len(trainids) + len(index_set) <= train_num:
                trainids += index_set
                train_scaffold_count += 1
            elif len(validids) + len(index_set) <= valid_num:
                validids += index_set
                val_scaffold_count += 1
            else:
                testids += index_set
                test_scaffold_count += 1

        print(f'Total scaffolds = {len(scaffolds)}, train scaffolds = {train_scaffold_count}, '
              f'val scaffolds = {val_scaffold_count}, test scaffolds = {test_scaffold_count}')

        trainset = self.id2data(dataset, trainids)
        validset = self.id2data(dataset, validids)
        testset = self.id2data(dataset, testids)

        return (trainset, validset, testset), (trainids, validids, testids)




