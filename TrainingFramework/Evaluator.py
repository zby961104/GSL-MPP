from TrainingFramework.Metrics import *
import time
import numpy as np


class GslMolEvaluator(object):
    def __init__(self, opt):
        super(GslMolEvaluator, self).__init__()
        self.opt = opt

    def eval(self, loader, eval_idx, model, metrics):
        model.eval()
        All_answer = []
        All_label = []
        for i in range(self.opt.args['TaskNum']):
            All_answer.append([])
            All_label.append([])

        for data in loader:
            data = data.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            Label = data.y
            # Label: [batch, task, 1]

            Label = Label.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))  # [batch, task, 1]
            # Label = Label.squeeze(-1)  # [batch, task]
            Label = Label[eval_idx].t()  # [task, batch]

            if self.opt.args['Model'] == 'GslMol':
                output, adj, init_node_vce = model(data)  # [batch, TaskNum * ClassNum]
            else:
                output = model(data)

            output = output[eval_idx]

            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:,
                                  i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                # [batch_size, ClassNum]
                cur_task_label = Label[i]  # [batch_size]
                # print(cur_task_label.size())

                cur_task_cur_batch_valid_labels = []
                cur_task_cur_batch_valid_answers = []
                for j in range(len(cur_task_label)):
                    l = cur_task_label[j]
                    if l == -1:
                        continue
                    else:
                        cur_task_cur_batch_valid_labels.append(l.item())
                        cur_task_cur_batch_valid_answers.append(cur_task_output[j].tolist())



                for ii, item in enumerate(cur_task_cur_batch_valid_labels):
                    All_label[i].append(item)
                for ii, item in enumerate(cur_task_cur_batch_valid_answers):
                    All_answer[i].append(item)

        if self.opt.args['ExpName'] in ['QM9', 'QM7']:
            All_label, All_answer = self.DeNormalized(All_label, All_answer)

        scores = {}
        All_metrics = []
        for i in range(self.opt.args['TaskNum']):
            # for each task, the All_label and All_answer contains the samples of which labels are not missing
            All_metrics.append([])
            label = All_label[i]
            answer = All_answer[i]
            assert len(label) == len(answer)
            for metric in metrics:
                result = metric.compute(answer, label)
                All_metrics[i].append(result)
                # if multitask, then print the results of each tasks.
                if self.opt.args['TaskNum'] != 1:
                    print("The value of metric", metric.name, "in task", i, 'is: ', result)
        average = t.Tensor(All_metrics).mean(dim = 0)  # dim 0 is the multitask dim.
        # the length of average is metrics num

        for i in range(len(metrics)):
            scores.update({metrics[i].name: average[i].item()})
            print("The average value of metric", metrics[i].name, "is: ", average[i].item())

        model.train()
        return scores

    def DeNormalized(self, labels, scores):
        z_mean = np.load(self.opt.args['TrialPath']+'ValueMean.npy')
        z_var = np.load(self.opt.args['TrialPath']+'ValueVar.npy')
        Labels = np.array(labels).T
        Scores = np.array(scores).T
        print(f"Labels.shape:{Labels.shape}")
        print(f"means.shape:{z_mean.shape}")
        print(f"var.shape:{z_var.shape}")
        Labels = Labels * np.sqrt(z_var) + z_mean
        Scores = Scores * np.sqrt(z_var) + z_mean
        return Labels.T.tolist(), Scores.T.tolist()



