import numpy as np
import pandas as pd

# from utils import Args
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class GraphRepresentation:
    def __init__(self, args):
        self.args = args
        pass

    def transform(self, x, edges=None):
        if not edges:
            edges = self.edges
        nmi = self._batch_NMI(x)
        adjacency = np.einsum('ijk, jk -> ijk', nmi, edges)
        return adjacency

    def fit(self, X, y, rank=100):
        y = y.squeeze()

        NMI_train = self._batch_NMI(X)
        NMI_left = NMI_train[np.where(y == 0)]
        NMI_right = NMI_train[np.where(y == 1)]

        distance = self._distance(NMI_right, NMI_left)

        edges = self._edges(distance, rank)
        self.edges = edges

        adjecency_train = np.einsum('ijk, jk -> ijk', NMI_train, edges)

        return adjecency_train

    def _edges(self, distance, rank):
        distance = np.asarray(distance)

        i_below = np.tril_indices(62, -1)
        distance[i_below] = 0

        buffer = np.ndarray.flatten(distance)
        mins = np.sort(buffer)[-rank]

        distance[distance >= mins] = 1
        distance[distance < mins] = 0

        distance[i_below] = distance.T[i_below]

        return distance

    def _distance(self, X, Y):
        A = np.empty((self.args.nodes, self.args.nodes))
        for i in range(self.args.nodes):
            for j in range(self.args.nodes):
                if j>i:
                    A[i, j] = wasserstein_distance(X[:, i, j], Y[:, i, j])
        i_below = np.tril_indices(62, -1)
        A[i_below] = A.T[i_below]
        return A

    def _batch_NMI(self, dataset):

        myEntropy = MyEntropy(dataset, self.args)
        executor = DataLoader(dataset=myEntropy, batch_size=128, shuffle=False,
                              num_workers=5)

        NMI = np.empty((0, self.args.nodes, self.args.nodes))
        for a in tqdm(executor):
            a = a
            NMI = np.vstack((NMI, a))

        self.nmi = NMI
        return NMI


class MyEntropy(Dataset):
    def __init__(self, x, args):
        super().__init__()
        self.x = x
        self.args = args

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        a = self._trial_NMI(x)
        return a

    def _trial_NMI(self, trial):
        NMIs = np.ones(shape=(self.args.nodes, self.args.nodes))
        for r in range(self.args.nodes):
            for c in range(self.args.nodes):
                if c > r:
                    X = trial[r]
                    Y = trial[c]
                    NMI = self._channel_NMI(X, Y)
                    NMIs[r,c] = NMI
        i_below = np.tril_indices(62, -1)
        NMIs[i_below] = NMIs.T[i_below]
        return NMIs

    def _channel_NMI(self, X, Y):
        c_XY = np.histogram2d(X, Y)[0]
        c_X = np.histogram(X)[0]
        c_Y = np.histogram(Y)[0]

        H_X = self._NEentropy(c_X)
        H_Y = self._NEentropy(c_Y)
        H_XY = self._NEentropy(c_XY)

        MI = H_X + H_Y - H_XY

        normMI = MI / np.sqrt((H_X*H_Y))
        return normMI

    def _NEentropy(self, c):
        c = c / float(np.sum(c))
        c = c[np.nonzero(c)]

        H = -sum(c * np.log2(c))
        return H


if __name__ == '__main__':
    # size must be b, [c, T]
    data = np.load('dataset/subj_1_data.npy', allow_pickle=True).transpose((0, 2, 1))
    label = np.load('dataset/subj_1_label.npy')
    #
    args = Args().get_args()
    mi = GraphRepresentation(args)
    #
    # arr = np.random.randint(low=0, high=2, size=(62,62))
    # randx = np.random.randint(0,19, (1, 62, 200))
    #
    # a = mi.transform(randx, arr)
    #
    edges = mi.fit(data[:50], label[:50], 50)



    pass