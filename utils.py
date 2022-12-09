import argparse

import numpy as np
from moabb.datasets import Lee2019_MI, Cho2017
from moabb.paradigms import MotorImagery as MI
import scipy.io
from scipy.signal import resample
from sklearn.model_selection import KFold
# from NMI import GraphRepresentation
import torch
from torch.utils.data import Dataset

from einops.layers.torch import Rearrange
from einops import repeat


class Args:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_args(self):
        if self.dataset == 'lee':
            return self._lee_args()
        if self.dataset == 'cho':
            return self._cho_args()

    def _lee_args(self):
        pars = argparse.ArgumentParser()

        #  Data Attributes:
        pars.add_argument('--subject_id', default=0, type=int)
        pars.add_argument('--k_fold', default=0, type=int)
        pars.add_argument('--nodes', default=62, type=int)

        #  Model Params
        pars.add_argument('--nhead', default=10, type=int)
        pars.add_argument('--d_model', default=200, type=int)
        pars.add_argument('--num_layers', default=1, type=int)
        pars.add_argument('--nclass', default=2, type=int)

        # Hyperparameter
        pars.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
        pars.add_argument('--lrate', default=0.005, type=float)
        pars.add_argument('--iteration', default=100, type=int)
        pars.add_argument('--test_len', default=0, type=int)
        pars.add_argument('--batch', default=32, type=int)
        pars.add_argument('--num_workers', default=1, type=int)
        pars.add_argument('--dataset', default='lee', type=str)
        pars.add_argument('--window', default=200, type=int)
        pars.add_argument('--rank', default=100, type=int)

        argset = pars.parse_args()
        return argset

    def _cho_args(self):
        pars = argparse.ArgumentParser()

        #  Data Attributes:
        pars.add_argument('--subject_id', default=0, type=int)
        pars.add_argument('--k_fold', default=0, type=int)
        pars.add_argument('--nodes', default=64, type=int)

        #  Model Params
        pars.add_argument('--nhead', default=10, type=int)
        pars.add_argument('--d_model', default=200, type=int)
        pars.add_argument('--num_layers', default=1, type=int)
        pars.add_argument('--nclass', default=2, type=int)

        # Hyperparameter
        pars.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
        pars.add_argument('--lrate', default=0.005, type=float)
        pars.add_argument('--iteration', default=100, type=int)
        pars.add_argument('--test_len', default=0, type=int)
        pars.add_argument('--batch', default=32, type=int)
        pars.add_argument('--num_workers', default=1, type=int)
        pars.add_argument('--dataset', default='cho', type=str)
        pars.add_argument('--window', default=200, type=int)
        pars.add_argument('--rank', default=100, type=int)

        argset = pars.parse_args()
        return argset


class LoadData:
    def __init__(self, args):
        self.args = args

    def run(self):
        if self.args.dataset == 'lee':
            GetData_Lee(self.args).getSubjectData(self.args.subject_id)
        if self.args.dataset == 'cho':
            GetData_Cho(self.args).getSubjectData(self.args.subject_id)


class GetData_Cho:
    def __init__(self, args):
        self.args = args
        self._source = Cho2017()
        self._downloader = MI(n_classes=2, tmin=0, tmax=4)

    def getSubjectData(self, subject):
        x, y = self.preprocessing(subject)
        np.save(f'dataset_{self.args.dataset}/subj_{subject}_data', x)
        np.save(f'dataset_{self.args.dataset}/subj_{subject}_label', y)

    def preprocessing(self, subject):
        x, y, _ = self._downloader.get_data(dataset=self._source, subjects=[subject])
        _, y = np.unique(y, return_inverse=True)
        y = y[:, np.newaxis]

        xs = resample(x, 1000, axis=2)
        for t in range(x.shape[0]):
            m = np.mean(xs[t])
            std = np.std(xs[t])
            xs[t] = (xs[t] - m) / std
        return xs, y




class GetData_Lee:
    def __init__(self, args):
        self._source = Lee2019_MI(test_run=False)
        self.args = args

    def getAllData(self):
        subject_list = list(np.linspace(1, 54, 54, dtype=int))
        for s in subject_list:
            self.getSubjectData(s)

    def getSubjectData(self, subject_id):
        self._accessData(subject_id)
        np.save(f'dataset_{self.args.dataset}/subj_{subject_id}_data', self._x)
        np.save(f'dataset_{self.args.dataset}/subj_{subject_id}_label', self._y)

    def _accessData(self, subject_id):
        dir = self._source.data_path(subject_id)
        X1 = scipy.io.loadmat(dir[0])
        x1, y1 = self._loadData(X1)

        X2 = scipy.io.loadmat(dir[0])
        x2, y2 = self._loadData(X2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2)) - 1

        x = torch.tensor(x).unfold(step=self.args.d_model // 2, size=self.args.d_model, dimension=2)
        self._x = np.asarray(Rearrange('b s c w -> (b s) c w', w=self.args.d_model)(torch.transpose(x, 2, 1)))
        self._y = np.asarray(repeat(y, 'h c -> (h c r)', r=self._x.shape[0]//y.shape[0])[:, np.newaxis])

    def _loadData(self, X):
        X1 = X['EEG_MI_train']
        X2 = X['EEG_MI_test']

        x1 = X1['smt'].item()
        y1 = X1['y_dec'].item()

        x2 = X2['smt'].item()
        y2 = X2['y_dec'].item()

        x = np.concatenate((x1, x2), axis=1).transpose((1, 2, 0))
        y = np.concatenate((y1, y2), axis=1)

        xs = resample(x, 1000, axis=2)
        for t in range(x.shape[0]):
            m = np.mean(xs[t])
            std = np.std(xs[t])
            xs[t] = (xs[t] - m) / std

        return xs, y.transpose((1, 0))

    def _downloadData(self, subject):
        x, y, _ = self._source.download(subject_list=[subject])


class CrossValidation:
    def __init__(self, args):
        self.fold = args.k_fold
        self.folders = KFold(n_splits=5, shuffle=True, random_state=1000)
        self.x = np.load(f'dataset_{args.dataset}/subj_{args.subject_id}_data.npy')
        self.y = np.load(f'dataset_{args.dataset}/subj_{args.subject_id}_label.npy')
        self._indexing()

    def getData(self):
        return self.x[self._trainidx], self.y[self._trainidx], self.x[self._testidx], self.y[self._testidx]

    def _indexing(self):
        test_idx = []
        idx = list(np.linspace(start=0, stop=self.x.shape[0] - 1,
                               num=self.x.shape[0], dtype=int))
        for trainidx, testidx in self.folders.split(self.x):
            test_idx.append(testidx)

        self._testidx = test_idx[self.fold]
        self._trainidx = np.array([t for t in idx if t not in self._testidx])


class Iterator(Dataset):
    def __init__(self, data, label, adjacency):
        super(Iterator, self).__init__()
        self.x = data
        self.y = label
        self.adjacency = adjacency

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        x[np.isnan(x)] = 0
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.adjacency[i], dtype=torch.float32)

        return x, y, a


if __name__ == '__main__':
    args = Args('lee').get_args()
    args.subject_id = 1
    LoadData(args).run()
    pass
