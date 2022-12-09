import numpy as np

from utils import Args, CrossValidation, Iterator, LoadData
from trainers import Train
from torch.utils.data import DataLoader
from NMI import GraphRepresentation


class Main:
    def __init__(self, dataset):
        self.arg = Args(dataset).get_args()

    def fit(self, subject, fold):
        self.arg.subject_id = subject
        self.arg.k_fold = fold

        LoadData(self.arg).run()
        dataset = CrossValidation(args=self.arg)
        x_train, y_train, x_test, y_test = dataset.getData()

        self.arg.test_len = x_test.shape[0]

        graph = GraphRepresentation(self.arg)
        adj_train, edges = graph.fit(x_train, y_train, rank=200)
        adj_test = graph.transform(x_test)

        np.save(f'adjacency_{self.arg.dataset}/adj_train_{subject}_{fold}', adj_train)
        np.save(f'adjacency_{self.arg.dataset}/adj_test_{subject}_{fold}', adj_test)
        np.save(f'adjacency_{self.arg.dataset}/edges_{subject}_{fold}', edges)

        adj_train = np.load(f'adjacency_{self.arg.dataset}/adj_train_{subject}_{fold}.npy')
        adj_test = np.load(f'adjacency_{self.arg.dataset}/adj_test_{subject}_{fold}.npy')

        train_iters = Iterator(x_train, y_train, adj_train)
        test_iters = Iterator(x_test, y_test, adj_test)

        train_load = DataLoader(dataset=train_iters, batch_size=self.arg.batch, shuffle=True,
                                num_workers=self.arg.num_workers)
        test_load = DataLoader(dataset=test_iters, batch_size=self.arg.batch * 4, shuffle=False,
                               num_workers=self.arg.num_workers)

        training = Train(arg=self.arg)
        acc = training.fit(train_load, test_load)

        report = f'\n{self.arg.dataset},{subject},{fold},{acc}'
        write_text = open('sample.txt', "a")
        write_text.write(report)


if __name__ == '__main__':
    framework = Main('lee')
    framework.fit(subject=1, fold=0)
