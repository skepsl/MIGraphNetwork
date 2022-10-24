import numpy as np

from utils import Args, CrossValidation, Iterator
from trainers import Train
from torch.utils.data import DataLoader
from NMI import GraphRepresentation


class Main:
    def __init__(self):
        self.arg = Args().get_args()

    def fit(self, subject, fold):
        self.arg.subject_id = subject
        self.arg.k_fold = fold

        dataset = CrossValidation(subject_id=self.arg.subject_id, fold_id=self.arg.k_fold)
        x_train, y_train, x_test, y_test = dataset.getData()

        # x_train, y_train, x_test, y_test = x_train[:50], y_train[:50], x_test[:50], y_test[:50]

        self.arg.test_len = x_test.shape[0]

        graph = GraphRepresentation(self.arg)
        adj_train = graph.fit(x_train, y_train, rank=100)
        adj_test = graph.transform(x_test)

        np.save('adj_train', adj_train)
        np.save('adj_test', adj_test)

        adj_train = np.load('adj_train.npy')
        adj_test = np.load('adj_test.npy')

        train_iters = Iterator(x_train, y_train, adj_train)
        test_iters = Iterator(x_test, y_test, adj_test)

        train_load = DataLoader(dataset=train_iters, batch_size=self.arg.batch, shuffle=True,
                                num_workers=self.arg.num_workers)
        test_load = DataLoader(dataset=test_iters, batch_size=self.arg.batch * 4, shuffle=False,
                               num_workers=self.arg.num_workers)

        training = Train(arg=self.arg)
        training.fit(train_load, test_load)


if __name__ == '__main__':
    framework = Main()
    framework.fit(subject=3, fold=1)
