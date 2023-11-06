import json
import os
import torch
import random
from TrainingFramework.Dataset import MolDatasetCreator
from TrainingFramework.Evaluator import *
from TrainingFramework.Scheduler import *
from TrainingFramework.Initializer import *

import torch.optim as optim
from Models.GslMol.molecule_gsl_model import MolRelationNet


class Saver(object):
    # Module that package the saving functions
    def __init__(self):
        super(Saver, self).__init__()
        #self.ckpt_state = {}

    def SaveContext(self, context_add, context_obj):
        # if something can be summarized as a dict {}
        # then using SaveContext function to save it into a json file.
        with open(context_add, 'w') as f:
            json.dump(context_obj, f)

    def LoadContext(self, context_add):
        # Using LoadContext to load a json file to a dict {}.
        with open(context_add, 'r') as f:
            obj = json.load(f)
        return obj


class Configs(object):
    def __init__(self, ParamList):
        # initiale a Config object with given paramlist
        super(Configs, self).__init__()
        self.args = {}
        for param in ParamList.keys():
            self.set_args(param, ParamList.get(param))

    def set_args(self, argname, value):
        if argname in self.args:
            print("Arg", argname, "is updated.")
            self.args[argname] = value
        else:
            print('Arg', argname, 'is added.')
            self.args.update({argname: value})

    def get_args(self, argname, value):
        if argname in self.args:
            return self.args[argname]
        else:
            return value


class ControllerStatusSaver(object):
    # Package functions for saving and loading status of a controller into/from a file.
    # In a ControllerStatusSaver, it maintains three global variable:
    # self.args: args
    # self.saver: Saver() object for file saving and loading.
    # self.Addr: The Addr to save the status of the controller.

    def __init__(self, args, ControllerType, Addr=None, restart=False):
        super(ControllerStatusSaver, self).__init__()
        self.saver = Saver()
        self.args = args

        if ControllerType == 'ExperimentProcessController':
            self.Addr = self.args['TrialPath'] + 'ExperimentProcessControllerStatus/'
        elif ControllerType == 'ConfigController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        elif ControllerType == 'EarlyStopController':
            self.Addr = self.args['SaveDir'] + 'EarlyStopControllerStatus/'
        elif ControllerType == 'Trainer':
            self.Addr = self.args['SaveDir'] + 'TrainerStatus/'
        elif ControllerType == 'CkptController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        else:
            if Addr:
                self.Addr = Addr
            else:
                raise KeyError(
                        'Wrong ControllerType given.'
                )
        self.CheckAddr()

        if restart:
            self.DeleteFilesInDir(self.Addr)

    def DeleteFilesInDir(self, addr):
        del_list = os.listdir(addr)
        for f in del_list:
            file_addr = addr + f
            os.remove(file_addr)

    def CheckAddr(self):
        if not os.path.exists(self.Addr):
            os.mkdir(self.Addr)

    def SaveStatus(self, status, restart=False):

        next_cnt = self.CountFileNames(self.Addr)
        if next_cnt != 0:
            assert self.LastFileName(self.Addr) == str(next_cnt-1)
            file_name = self.Addr + str(next_cnt)
        else:
            file_name = self.Addr + '0'
        self.saver.SaveContext(file_name, status)


    def LoadStatus(self, status_idx=None):
        # if the index is not given, then find the last file from the folder. the last file is the file to be loaded.
        # otherwise, the file of the given index is to be loaded.
        if not status_idx:
            file_name = self.Addr + self.LastFileName(self.Addr)
        else:
            file_name = self.Addr + str(status_idx)

        # if no file is to be loaded, then return None.
        # (e.g. empty in the folder or the given index not exists)
        if os.path.exists(file_name):
            return self.saver.LoadContext(file_name)
        else:
            return None

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        return last_file

    def CountFileNames(self, Addr):
        dir_files = os.listdir(Addr)
        return len(dir_files)

##########################################################################################################


class EarlyStopController(object):
    # A module used to control the early stop part of the experimental progress.
    # It maintains the result of each epoch, max results, count of worse results
    # and to make decision whether the training progress should be early stopped.
    def __init__(self, opt):
        super(EarlyStopController, self).__init__()
        self.opt = opt
        # params coming from the opt are constant during the training progress of THIS opt
        self.MetricName = opt.args['MainMetric']
        self.LowerThanMaxLimit = opt.args['LowerThanMaxLimit']
        self.DecreasingLimit = opt.args['DecreasingLimit']
        # Other params are the running status of the EarlyStopController that should be saved and loaded by files.
        # initial MaxResult
        if self.opt.args['ClassNum'] == 1:
            self.MaxResult = 9e8
        else:
            self.MaxResult = 0

        self.MaxResultModelIdx = 0
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx, testscore=None):
        # Make decision whether the training progress should be stopped.
        # When the current result is better than the MaxResult, then update thre MaxResult.
        # When the current result is worse that the MaxResult, then start to count.
        # When the num of epochs that the result is worse than the MaxResult exceed the LowerThanMaxLimit threshold, then stop
        # And when the result is persistently getting worse for over DecreasingLimit epochs, then stop.

        # score is the current Validation Result
        # ckpt_idx is the ckpt index
        # testscore is the result of the current model on the test set.

        MainScore = score[self.MetricName]
        if testscore:
            MainTestScore = testscore[self.MetricName]
        else:
            MainTestScore = None
        self.TestResult.append(MainTestScore)

        if self.opt.args['ClassNum'] != 1:
            # Classification, the larger the better
            if MainScore > self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all counts reset to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore < self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore
        else:
            # Regression, the lower the better
            if MainScore < self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all set to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore > self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        return self.MaxResult, self.MaxResultModelIdx, self.TestResult[self.MaxResultModelIdx]

    def GetControllerStatus(self):
        status = {
            'MaxResult': self.MaxResult,
            'MaxResultModelIdx': self.MaxResultModelIdx,
            'LastResult': self.LastResult,
            'LowerThanMaxNum': self.LowerThanMaxNum,
            'DecreasingNum': self.DecreasingNum,
            'TestResult': self.TestResult
        }
        return status

    def SetControllerStatus(self, status):
        self.MaxResult = status['MaxResult']
        self.MaxResultModelIdx = status['MaxResultModelIdx']
        self.LastResult = status['LastResult']
        self.LowerThanMaxNum = status['LowerThanMaxNum']
        self.DecreasingNum = status['DecreasingNum']
        self.TestResult = status['TestResult']

##########################################################################################################


class GreedyConfigController(object):
    # Here the basic greedy searching strategy is implemented.

    def __init__(self, BasicHyperparamList, AdjustableHyperparamList, SpecificHyperparamList=None):
        # Basic: Configs for training, not for HyperParam Searching
        # Adjustable: Configs for greedy searching, candidates.
        # Specific: Specific group of HyperParams, not for greedy searching.
        super(GreedyConfigController, self).__init__()
        self.BasicHyperparameterList = BasicHyperparamList
        self.HyperparameterList = AdjustableHyperparamList
        self.SpecificHyperparamList = SpecificHyperparamList
        self.opt = Configs(self.BasicHyperparameterList)
        self.MainMetric = self.BasicHyperparameterList['MainMetric']
        self.OnlySpecific = self.BasicHyperparameterList['OnlySpecific']

        # set the Trial Path for the experiment on this dataset.
        self.opt.set_args('TrialPath', self.opt.args['RootPath'] + self.opt.args['ExpName'] + '/')
        if not os.path.exists(self.opt.args['TrialPath']):
            os.mkdir(self.opt.args['TrialPath'])

        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'ConfigController')
        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()
            self.CheckSpecificHyperparamList(SpecificHyperparamList)
            self.OptInit(SpecificHyperparamList)


    def CheckSpecificHyperparamList(self, SpecificHyperparamList):
        firstkey = list(SpecificHyperparamList.keys())[0]
        SpecificChoiceNum = len(SpecificHyperparamList[firstkey])
        for key in SpecificHyperparamList.keys():
            assert SpecificChoiceNum == len(SpecificHyperparamList[key])

    def OptInit(self, SpecificHyperparamList):
        if SpecificHyperparamList:
            self.HyperparameterInit(self.SpecificHyperparamList)
        else:
            self.HyperparameterInit(self.HyperparameterList)

    def HyperparameterInit(self, paramlist):
        for param in paramlist.keys():
            self.opt.set_args(param, paramlist.get(param)[0])
        # initially, the hyperparameters are set to be the first value of their candidate lists each.

    def GetOpts(self):
        self.opt.set_args('ExpDir', self.opt.args['TrialPath'] + 'exp' + str(self.exp_count) + '/')
        if not os.path.exists(self.opt.args['ExpDir']):
            os.mkdir(self.opt.args['ExpDir'])
        return self.opt

    def AdjustParams(self):
        # Adjust the hyperparameters by greedy search.
        # The return is the end flag

        # if the Specific Hyperparam List is given, then set the opts as the param group in SpecificParamList
        if self.SpecificHyperparamList:
            keys = self.SpecificHyperparamList.keys()
            if self.exp_count < len(self.SpecificHyperparamList.get(list(keys)[0])):
                for param in self.SpecificHyperparamList.keys():
                    self.opt.set_args(param, self.SpecificHyperparamList.get(param)[self.exp_count])
                return False
            elif self.exp_count == len(self.SpecificHyperparamList.get(list(keys)[0])):
                if self.OnlySpecific:
                    return True
                else:
                    self.HyperparameterInit(self.HyperparameterList)
                    self.result = []
                    return False

        # After trying the given specific params, using greedy search in the AdjustableParamList(HyperParameterList).
        ParamNames = list(self.HyperparameterList.keys())
        cur_param_name = ParamNames[self.parampointer]           # key, string
        cur_param = self.HyperparameterList[cur_param_name]      # list of values
        if self.paramvaluepointer < len(cur_param):
            # set the config
            cur_value = cur_param[self.paramvaluepointer]        # value
            self.opt.set_args(cur_param_name, cur_value)
            self.paramvaluepointer += 1
        else:
            # choose the best param value based on the results.
            assert len(self.result) == len(cur_param)

            if self.opt.args['ClassNum'] == 1:
                best_metric = min(self.result)
            else:
                best_metric = max(self.result)

            loc = self.result.index(best_metric)
            self.result = []
            self.result.append(best_metric)                      # best_metric is obtained by configs: {paraml:[loc], paraml+1:[0]}
                                                                 # so that we don't need to test the choice of paraml+1:[0]
                                                                 # just use the result tested when adjusting paraml.
            cur_param_best_value = cur_param[loc]
            self.opt.set_args(cur_param_name, cur_param_best_value)
            self.parampointer += 1
            self.paramvaluepointer = 1                           # test from paraml+1:[1]

            if self.parampointer < len(ParamNames):
                # set the config
                cur_param_name = ParamNames[self.parampointer]
                cur_param = self.HyperparameterList[cur_param_name]
                cur_value = cur_param[self.paramvaluepointer]
                self.opt.set_args(cur_param_name, cur_value)
                self.paramvaluepointer += 1
                return False
            else:
                return True



    def StoreResults(self, score):
        self.result.append(score)

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        # running status of the ConfigController.
        self.exp_count = 0
        self.parampointer = 0
        self.paramvaluepointer = 1
        # two pointers indicates the param and its value that next experiment should use.
        self.result = []

    def GetControllerStatus(self):
        self.status = {
            'exp_count': self.exp_count,
            'parampointer': self.parampointer,
            'paramvaluepointer': self.paramvaluepointer,
            'result': self.result,
            'next_opt_args': self.opt.args
        }

    def SetControllerStatus(self, status):
        self.exp_count = status['exp_count']
        self.parampointer = status['parampointer']
        self.paramvaluepointer = status['paramvaluepointer']
        self.result = status['result']
        self.opt.args = status['next_opt_args']
        #print("Config Controller has been loaded. Experiments continue.")


##########################################################################################################

class CkptController(object):
    # controller to deal with the check point after each epoch training.
    # A ckpt happens on the end of a training epoch
    # including following information:
    # model: the model after current epoch training
    # optimizer: the optimizer after current epoch training
    # epoch: the epoch number
    # These information should be saved and be loaded and maintained for latter training.
    # scores: the result on valid set
    # testscores: the result on test set
    # These informations should be saved as files and not need to maintain and reload.
    # ckpt_count: The status that should be maintained during training.

    # The status of the ckpt controller is only the ckpt_count and the ckpt_name
    # and other informations are not needed to be maintained in the controller.
    # they should be saved when calling CkptProcessing function, and loaded when calling LoadCkpt function.
    def __init__(self, opt):
        super(CkptController, self).__init__()
        self.opt = opt
        self.ckpt_count = 0
        self.saver = Saver()
        #

    def CkptProcessing(self, model, optimizer, epoch, scores, testscores):
        # Saving the check point to a file.
        ckpt_name = self.opt.args['SaveDir'] + 'model/model_optimizer_epoch' + str(self.ckpt_count)
        ckpt = {
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch
        }
        self.SaveCkpt(ckpt, ckpt_name)
        # Saving the result to a file.
        valid_result_file_name = self.opt.args['SaveDir'] + 'result' + str(self.ckpt_count) + 'valid.json'
        self.saver.SaveContext(valid_result_file_name, scores)
        test_result_file_name = self.opt.args['SaveDir'] + 'result' + str(self.ckpt_count) + 'test.json'
        self.saver.SaveContext(test_result_file_name, testscores)
        print('Results saved.')

        self.ckpt_count += 1

    def SaveCkpt(self, ckpt, ckpt_name):
        t.save(ckpt, ckpt_name)
        print('Model Saved.')

    def LoadCkpt(self):
        last_model_ckpt = self.FindLastCkpt()
        if last_model_ckpt:
            ckpt = t.load(os.path.join(self.opt.args['ModelDir'], last_model_ckpt))
            model = ckpt['model']
            optimizer = ckpt['optimizer']
            epoch = ckpt['epoch']
            return model, optimizer, epoch
        else:
            return None, None, None

    def FindLastCkpt(self):
        dir_files = os.listdir(self.opt.args['ModelDir'])  # list of the ckpt files.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.opt.args['ModelDir'],x)))
            last_model_ckpt = dir_files[-1] # find the latest ckpt file.
            return last_model_ckpt
        else:
            return None

    def GetControllerStatus(self):
        status = {
            'ckpt_count': self.ckpt_count
        }
        return status

    def SetControllerStatus(self, status):
        self.ckpt_count = status['ckpt_count']

##########################################################################################################


class ExperimentProcessController(object):
    # Module to control the entire experimental process.
    def __init__(self, ExpOptions, Params):
        # ExpOptions: Options to set the ExperimentProcessController
        # Params: Params for the experiment. i.g. the three ParamLists of greedy search.
        super(ExperimentProcessController, self).__init__()

        self.ExpOptions = ExpOptions
        self.search = self.ExpOptions['Search']
        self.seedperopt = self.ExpOptions['SeedPerOpt']

        # process the params based on different searching methods, determined by the ExpOptions
        if self.search == 'greedy' or self.search == 'grid':
            self.BasicParamList, self.AdjustableParamList, self.SpecificParamList = Params

        self.ConfigControllersList = {
            'greedy': GreedyConfigController
        }

        # os.environ['CUDA_VISIBLE_DEVICES'] = self.BasicParamList['CUDA_VISIBLE_DEVICES']
        self.configcontroller = self.ConfigControllersList[self.search](self.BasicParamList, self.AdjustableParamList, self.SpecificParamList)

        self.controllerstatussaver = ControllerStatusSaver(self.configcontroller.opt.args,'ExperimentProcessController')

        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()
        # The status: cur_opt_results, opt_results and i have been set, either initialized or loaded from files.

    def ExperimentStart(self):

        end_flag = False

        while not end_flag:
            opt = self.configcontroller.GetOpts()

            while self.i < self.seedperopt:
                self.CheckDirectories(opt, self.i)
                opt.set_args('split_seed', self.i + opt.args['SplitSeedBias'])

                print("The parameters of the current exp are: ")
                print(opt.args)

                trainer = RelationTrainer(opt)

                ckpt, value = trainer.TrainOneOpt()

                print(f"cur_opt_cur_seed_value: {value}")

                self.cur_opt_results.append(value)
                self.i += 1
                self.SaveStatusToFile()
                self.configcontroller.SaveStatusToFile()

            cur_opt_value = np.mean(self.cur_opt_results)     # the average result value of the current opt on self.seedperopt times running.
            self.opt_results.append(cur_opt_value)
            self.cur_opt_results = []                         # clear the buffer of current opt results.

            self.configcontroller.StoreResults(cur_opt_value)
            self.configcontroller.exp_count += 1
            end_flag = self.configcontroller.AdjustParams()
            self.i = 0

            if self.ExpOptions['SeedSearch']:
                end_flag = True

        print("Experiment Finished")
        print("The best averaged value of all opts is: ")
        if opt.args['ClassNum'] == 1:
            best_opt_result = min(self.opt_results)
            print(best_opt_result)
        else:
            best_opt_result = max(self.opt_results)
            print(best_opt_result)
        print("And the corresponding exp num is: ")
        print(self.opt_results.index(best_opt_result))

    def CheckDirectories(self, opt, i):
        opt.set_args('SaveDir', opt.args['ExpDir'] + str(i) + '/')
        if not os.path.exists(opt.args['SaveDir']):
            os.mkdir(opt.args['SaveDir'])

        opt.set_args('ModelDir', opt.args['SaveDir'] + 'model/')
        if not os.path.exists(opt.args['ModelDir']):
            os.mkdir(opt.args['ModelDir'])

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        self.cur_opt_results = []
        self.opt_results = []
        self.i = 0

    def GetControllerStatus(self):
        self.status = {
            'cur_opt_results': self.cur_opt_results,
            'opt_results': self.opt_results,
            'cur_i': self.i,
        }

    def SetControllerStatus(self, status):
        self.cur_opt_results = status['cur_opt_results']
        self.opt_results = status['opt_results']
        self.i = status['cur_i']
        assert self.i == len(self.cur_opt_results)


class RelationTrainer(object):

    def __init__(self, opt, KFold=False, no_eval_flag=False):
        super(RelationTrainer, self).__init__()
        self.opt = opt
        self.KFold = KFold
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        # self.device = t.device('cpu')

        self.no_eval_flag = no_eval_flag

        t.manual_seed(self.opt.args['TorchSeed'])
        np.random.seed(self.opt.args['TorchSeed'])
        random.seed(self.opt.args['TorchSeed'])
        os.environ['PYTHONHASHSEED'] = str(self.opt.args['TorchSeed'])
        t.cuda.manual_seed(self.opt.args['TorchSeed'])
        t.cuda.manual_seed_all(self.opt.args['TorchSeed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.BuildDataset()

        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildOptimizer()

        self.StartEpoch = 0

        self.lr_sch = self.BuildScheduler()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)

        # record the loss and auc to study the training process
        if 'log_loss' in opt.args.keys():
            self.log_loss = opt.args['log_loss']
        else:
            self.log_loss = False

        if self.log_loss:
            self.train_loss_list = []
            self.valid_result_list = []
            self.test_result_list = []

    def reinit(self):
        # self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        self.device = t.device('cpu')
        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildOptimizer()
        self.StartEpoch = 0

        self.lr_sch = self.BuildScheduler()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart = True)

    def BuildModel(self):
        if self.opt.args['Model'] == 'GslMol':
            net = MolRelationNet(
                self.opt, smiles_list=self.smiles_list
            ).to(self.device)
        else:
            raise NotImplementedError

        return net

    def BuildIniter(self):
        init_type = self.opt.args['WeightIniter']
        if init_type == 'Norm':
            self.initer = NormalInitializer(self.opt)

        elif init_type == 'XavierNorm':
            self.initer = XavierNormalInitializer()

        else:
            self.initer = None

    def BuildScheduler(self):
        if self.opt.args['Scheduler'] == 'EmptyLRScheduler':
            lr_sch = EmptyLRSchedular(self.optimizer, lr=10 ** -self.opt.args['lr'])

        elif self.opt.args['Scheduler'] == 'PolynomialDecayLR':
            # tot_updates = self.TrainsetLength * self.opt.args['MaxEpoch'] / self.opt.args['BatchSize']
            # warmup_updates = tot_updates / self.opt.args['WarmupRate']
            warmup_updates = self.opt.args['WarmupEpoch']
            tot_updates = self.opt.args['LRMaxEpoch']
            # warmup_updates = self.opt.args['WarmupUpdates']
            # tot_updates = self.opt.args['TotUpdeates']
            lr = 10 ** -self.opt.args['lr']
            end_lr = self.opt.args['EndLR']
            power = self.opt.args['Power']
            lr_sch = PolynomialDecayLR(self.optimizer, warmup_updates, tot_updates, lr, end_lr, power)

        elif self.opt.args['Scheduler'] == 'StepLR':
            step_size = self.opt.args['LRStep']
            gamma = self.opt.args['LRGamma']
            lr_sch = t.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        return lr_sch

    def BuildEvaluator(self):
        evaluator = GslMolEvaluator(self.opt)
        return evaluator

    def BuildDataset(self):

        moldatasetcreator = MolDatasetCreator(self.opt)
        all_set, idxs, self.weights, self.smiles_list = moldatasetcreator.CreateDatasets()

        # print(f"weights: {self.weights}")
        if len(self.opt.args['SplitRate']) == 2:
            (train_idx, valid_idx, test_idx) = idxs
        elif len(self.opt.args['SplitRate']) == 1:
            (train_idx, valid_idx) = idxs
            test_idx = []
        else:
            (train_idx) = idxs
            valid_idx = []
            test_idx = []

        self.batchsize = len(all_set)

        import torch_geometric as tg
        self.TrainsetLength = len(train_idx)

        self.loader = tg.loader.DataLoader(all_set, batch_size=self.batchsize, shuffle=False, num_workers=0,
                                           worker_init_fn=np.random.seed(8), pin_memory=True)
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

    def BuildOptimizer(self):
        if self.opt.args['Model'] == 'Graphormer':
            self.optimizer = optim.AdamW(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                                         weight_decay = 10 ** -self.opt.args['WeightDecay'])
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                                        weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildCriterion(self):
        if self.opt.args['ClassNum'] == 2:
            if self.opt.args['Weight']:
                self.criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction = 'mean').\
                                  to(self.device) for weight in self.weights]
            else:
                self.criterion = [nn.CrossEntropyLoss().\
                                      to(self.device) for i in range(self.opt.args['TaskNum'])]
        elif self.opt.args['ClassNum'] == 1:
            self.criterion = [nn.MSELoss().\
                                  to(self.device) for i in range(self.opt.args['TaskNum'])]


    def SaveModelCkpt(self, ckpt_idx):

        model = self.net
        optimizer = self.optimizer

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model' + str(ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer}
        t.save(ckpt, ckpt_name)
        print("Model Ckpt Saved!")

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result=None):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.opt.args['SaveDir'] + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def SaveTrainerStatus(self, epoch):
        self.GetControllerStatus(epoch)
        self.controllerstatussaver.SaveStatus(self.status)
        print("Trainer status saved!")

    def GetControllerStatus(self, epoch):
        if self.opt.args['ClassNum'] == 2:
            best_valid = self.BestValidAUC
            test_of_best_valid = self.TestAUCofBestValid


        elif self.opt.args['ClassNum'] == 1:
            best_valid = self.BestValidRMSE
            test_of_best_valid = self.TestRMSEofBestValid


        self.status = {
            'cur_epoch': epoch,
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    ########################################################

    def TrainOneOpt(self):
        print("Saving Current opt...")
        self.saver.SaveContext(self.opt.args['SaveDir'] + 'config.json', self.opt.args)

        print("Start Training...")
        epoch = self.StartEpoch
        stop_flag = 0

        if self.opt.args['ClassNum'] == 2:
            self.BestValidAUC = 0
            self.TestAUCofBestValid = 0

        elif self.opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 10e8
            self.TestRMSEofBestValid = 10e8

        if self.log_loss:
            self.train_loss_list = []
            self.valid_result_list = []
            self.test_result_list = []

        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                self.RemoveOtherCkpts(BestModel)
                self.SaveTrainerStopFlag()
                break

            self.TrainOneEpoch(self.net, self.loader, self.train_idx, self.valid_idx, self.test_idx, self.optimizer, self.criterion, self.evaluator)

            if not self.no_eval_flag:
                start_time = time.time()
                stop_flag = self.ValidOneTime(epoch, self.net)
                # print(f"Eval time: {time.time()-start_time}")
            else:
                self.FakeValidOneTime()
                try:
                    stop_flag = self.ReadStopFlagFile()
                except:
                    stop_flag = False

            self.SaveModelCkpt(epoch)
            self.SaveTrainerStatus(epoch)
            epoch += 1

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Stop Training.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        self.RemoveOtherCkpts(BestModel)

        if self.log_loss:
            train_loss_array = np.array(self.train_loss_list)
            valid_result_array = np.array(self.valid_result_list)
            test_result_array = np.array(self.test_result_list)
            log_save_dir = self.opt.args['SaveDir'] + 'train_process/'
            if not os.path.exists(log_save_dir):
                os.mkdir(log_save_dir)
            np.save(log_save_dir + 'train_loss', train_loss_array)
            np.save(log_save_dir + 'valid_result', valid_result_array)
            np.save(log_save_dir + 'test_result', test_result_array)

        return BestModel, MaxResult

    ################################################################################################################

    def TrainOneEpoch(self, model, loader, train_idx, valid_idx, test_idx, optimizer, criterion, evaluator):

        for data in loader:
            data.to(self.device)
            # print(data)
            # print("Here C")
            Label = data.y
            Label = Label.to(self.device)
            # Label = Label.squeeze(-1)       # [batch, task]
            Label = Label.t()               # [task, batch]

            if self.opt.args['Model'] == 'GslMol':
                output, adj, init_node_vec = model(data)
            else:
                output = model(data)

            loss = self.CalculateLoss(output[train_idx], Label[:, train_idx], criterion)
            if self.opt.args['adj_loss']:
                adj_loss_weight = self.opt.get_args('adj_loss_weight', 0.1)
                adj_loss = self.CalculateAdjLoss(adj, Label, train_idx) * self.opt.get_args('adj_loss_scale', 1)
                loss = adj_loss_weight * adj_loss + (1 - adj_loss_weight) * loss
            # if 'graph_learn_regularization' in self.opt.args.keys() and self.opt.args['graph_learn_regularization']:
            #     loss += self.add_graph_reg_loss(raw_adj, init_node_vec)

            loss.backward()

        # update the parameters
            optimizer.step()
            optimizer.zero_grad()

        # Print the loss
            if self.log_loss:
                self.train_loss_list.append(loss.item())

        self.lr_sch.step()

    def CalculateLoss(self, output, Label, criterion):
        loss = 0.0
        # print(f"output size: {output.size()}")
        # print(f"label size: {Label.size()}")
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.
                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    def add_graph_reg_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.opt.args['smoothness_ratio'] * torch.trace(
            torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = torch.ones(out_adj.size(-1)).to(self.device)
        graph_loss += -self.opt.args['degree_ratio'] * torch.mm(ones_vec.unsqueeze(0), torch.log(
            torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
        graph_loss += self.opt.args['sparsity_ratio'] * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def CalculateAdjLoss(self, adj, label, idx):
        if self.opt.args['ClassNum'] == 1:
            label = label.t()
            num_samples = label.size(0)
            label_i = label.repeat(1, num_samples)
            label_j = label_i.transpose(0, 1)
            diff = torch.abs(label_i - label_j)

            one = torch.ones_like(diff)
            zero = torch.zeros_like(diff)
            mask = torch.where(diff > self.opt.args['label_epsilon'], one, zero)
            num_ones = mask[idx, :][:, idx].sum()
            masked_adj = mask * adj
            masked_adj = masked_adj[idx, :][:, idx]

            adj_loss = (masked_adj * masked_adj).sum() / num_ones
        else:
            if self.opt.args['TaskNum'] == 1:
                label = label.t()
                label_edge = self.label2edge(label[idx])
                pred_edge = adj[idx, :][:, idx]
                adj_loss = F.mse_loss(pred_edge, label_edge)
            else:
                adj_loss = 0.0
                for i in range(self.opt.args['TaskNum']):
                    cur_task_label = label[i]
                    cur_adj = adj[i]
                    label_train = cur_task_label[idx]
                    label_edge = self.label2edge(label_train.unsqueeze(-1))
                    pred_edge = cur_adj[idx, :][:, idx]
                    valid_index = (label_edge != -1)
                    adj_loss += F.mse_loss(label_edge[valid_index], pred_edge[valid_index])

        return adj_loss

    def label2edge(self, label, mask_diag=False):

        # get size
        num_samples = label.size(0)
        # reshape
        label_i = label.repeat(1, num_samples)
        label_j = label_i.transpose(0, 1)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # if self.opt.args['rel_adj'] == 'dist':
        #     edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(0)).to(edge.device)
            edge = edge * diag_mask

        edge = edge / edge.sum(-1).unsqueeze(-1)

        return edge

    ################################################################################################################
    def ValidOneTime(self,epoch,net):
        if self.opt.args['ClassNum'] == 1:
            if self.opt.args['Splitter'] == None:
                print(f"Evaluate on Training set.")
                trainresult = self.evaluator.eval(self.loader, self.train_idx, net, [MAE(), RMSE()])

            print('Running on Valid set')
            result = self.evaluator.eval(self.loader, self.valid_idx, net, [MAE(), RMSE()])
            if self.log_loss:
                self.valid_result_list.append(result['RMSE'])
            if len(self.test_idx) != 0:
                print('Running on Test set')
                testresult = self.evaluator.eval(self.loader, self.test_idx, net, [MAE(), RMSE()])
                if self.log_loss:
                    self.test_result_list.append(testresult['RMSE'])

            if self.opt.args['Splitter'] == None:
                result = trainresult

            valid_result_rmse = result['RMSE']
            if len(self.test_idx) != 0:
                test_result_rmse = testresult['RMSE']
            else:
                test_result_rmse = None

            if valid_result_rmse < self.BestValidRMSE:
                self.BestValidRMSE = valid_result_rmse
                self.TestRMSEofBestValid = test_result_rmse
            print('Best Valid: ')
            print(self.BestValidRMSE)
            if len(self.test_idx) != 0:
                print('Best Test: ')
                print(self.TestRMSEofBestValid)

            self.SaveResultCkpt(epoch, valid_result_rmse, test_result_rmse)

            if len(self.test_idx) != 0:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

        else:
            if self.opt.args['Splitter'] == None:
                print(f"Evaluate on Training set.")
                trainresult = self.evaluator.eval(self.loader, self.train_idx, net, [AUC(), ACC()])

            print("Running on Valid set")
            result = self.evaluator.eval(self.loader, self.valid_idx, net, [AUC(), ACC()])
            if self.log_loss:
                self.valid_result_list.append(result['AUC'])
            if len(self.test_idx) != 0:
                print("running on Test set.")
                testresult = self.evaluator.eval(self.loader, self.test_idx, net, [AUC(), ACC()])
                if self.log_loss:
                    self.test_result_list.append(testresult['AUC'])

            if self.opt.args['Splitter'] == None:
                result = trainresult

            valid_result_auc = result['AUC']
            if len(self.test_idx) != 0:
                test_result_auc = testresult['AUC']
            else:
                test_result_auc = None

            if valid_result_auc > self.BestValidAUC:
                self.BestValidAUC = valid_result_auc
                self.TestAUCofBestValid = test_result_auc
            print('Best Valid: ')
            print(self.BestValidAUC)
            if len(self.test_idx) != 0:
                print('Best Test: ')
                print(self.TestAUCofBestValid)

            self.SaveResultCkpt(epoch, valid_result_auc, test_result_auc)

            if self.opt.args['Splitter'] == None:
                result = trainresult
            if len(self.test_idx) != 0:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag


    def FakeValidOneTime(self):
        for (ii, data) in enumerate(self.validloader):
            a = 1
        for (ii, data) in enumerate(self.testloader):
            a = 1


    def ReadStopFlagFile(self):
        stop_flag_file = self.opt.args['SaveDir'] + 'stop_flag.json'
        stop_flag = self.saver.LoadContext(stop_flag_file)
        return stop_flag

    def SaveTrainerStopFlag(self):
        trainer_stop_flag_file = self.opt.args['SaveDir'] + 'trainer_stop_flag.json'
        trainer_stop_flag = True
        self.saver.SaveContext(trainer_stop_flag_file, trainer_stop_flag)
        return

    def WeightInit(self):
        for param in self.net.parameters():
            self.initer.WeightInit(param)

    def RemoveOtherCkpts(self, bestmodel):
        if bestmodel == None:
            print(f"Ckpts will be deleted by Seperated Evaler.")
            return 0

        print(f"Deleting other ckpt models.")
        model_dir = self.opt.args['SaveDir'] + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.opt.args['SaveDir'] + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.opt.args['SaveDir'] + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # print(f"dir_files: {dir_files}")
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file



