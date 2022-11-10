import argparse

import numpy as np
from moabb.datasets import Lee2019_MI
import scipy.io
from scipy.signal import resample
from sklearn.model_selection import KFold
# from NMI import GraphRepresentation
import torch
from torch.utils.data import Dataset

from einops.layers.torch import Rearrange
from einops import repeat


class Args:
    def __init__(self):
        pass

    def get_args(self):
        return self._lee_args()

    def _lee_args(self):
        pars = argparse.ArgumentParser()

        #  Data Attributes:
        pars.add_argument('--subject_id', default=0, type=int)
        pars.add_argument('--k_fold', default=0, type=int)
        pars.add_argument('--nodes', default=62, type=int)
        pars.add_argument('--d_len', default=200, type=int)

        #  Model Params
        pars.add_argument('--nhead', default=10, type=int)
        pars.add_argument('--d_model', default=50, type=int)
        pars.add_argument('--num_layers', default=1, type=int)
        pars.add_argument('--nclass', default=2, type=int)

        # Hyperparameter
        pars.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
        pars.add_argument('--lrate', default=0.005, type=float)
        pars.add_argument('--iteration', default=50, type=int)
        pars.add_argument('--test_len', default=0, type=int)
        pars.add_argument('--batch', default=32, type=int)
        pars.add_argument('--num_workers', default=1, type=int)

        argset = pars.parse_args()
        return argset


class GetData:
    def __init__(self, args):
        self._source = Lee2019_MI(test_run=False)
        self.args = args

    def getAllData(self):
        subject_list = list(np.linspace(1, 54, 54, dtype=int))
        for s in subject_list:
            self.getSubjectData(s)

    def getSubjectData(self, subject_id):
        self._accessData(subject_id)
        np.save(f'dataset/subj_{subject_id}_data', self._x)
        np.save(f'dataset/subj_{subject_id}_label', self._y)

    def _accessData(self, subject_id):
        dir = self._source.data_path(subject_id)
        X1 = scipy.io.loadmat(dir[0])
        x1, y1 = self._loadData(X1)

        X2 = scipy.io.loadmat(dir[0])
        x2, y2 = self._loadData(X2)

        self._x = np.concatenate((x1, x2))
        self._y = np.concatenate((y1, y2)) - 1

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

        x, y = self._slidingWindow(xs.transpose((0, 2, 1)), y.transpose((1, 0)))
        return x, y

    def _slidingWindow(self, x, y):
        x = torch.tensor(x).unfold(step=self.args.d_model, size=self.args.d_model, dimension=1)

        y = repeat(y, 'h w -> h w s', s=x.shape[1])
        x = Rearrange('b w c s -> (b w) c s')(x)

        x = np.asarray(x)
        y = np.asarray(y).reshape((y.shape[0]*y.shape[2], 1))
        return x, y

    def _downloadData(self, subject_id):
        x, y, _ = self._source.download(subject_list=[subject_id])


class CrossValidation:
    def __init__(self, subject_id, fold_id):
        self.fold = fold_id - 1
        self.folders = KFold(n_splits=5, shuffle=True, random_state=1000)
        self.x = np.load(f'dataset/subj_{subject_id}_data.npy')
        self.y = np.load(f'dataset/subj_{subject_id}_label.npy')
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
        self.args = Args().get_args()

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
    args = Args().get_args()
    GetData(args).getSubjectData(1)
    pass
