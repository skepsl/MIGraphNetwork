import numpy as np

from utils import Args, CrossValidation, Iterator, GetData
from trainers import Train
from torch.utils.data import DataLoader
from NMI import GraphRepresentation


class Main:
    def __init__(self):
        self.arg = Args().get_args()
        

    def fit(self, subject, fold):
        GetData(self.arg).getSubjectData(subject)
        self.arg.subject_id = subject
        self.arg.k_fold = fold

        dataset = CrossValidation(subject_id=self.arg.subject_id, fold_id=self.arg.k_fold)
        x_train, y_train, x_test, y_test = dataset.getData()

        self.arg.test_len = x_test.shape[0]

        graph = GraphRepresentation(self.arg)
        adj_train = graph.fit(x_train, y_train, rank=100)
        adj_test = graph.transform(x_test)

        # Save for re-run later
        np.save(f'adjacency/adj_train_{subject}_{fold}', adj_train)
        np.save(f'adjacency/adj_test_{subject}_{fold}', adj_test)
        adj_train = np.load(f'adjacency/adj_train_{subject}_{fold}.npy')
        adj_test = np.load(f'adjacency/adj_test_{subject}_{fold}.npy')

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

    # Junhee
    subjects = list(np.linspace(1, 10, 10, dtype=int))
    
    # Seunghoon
    subjects = list(np.linspace(11, 32, 22, dtype=int))
    
    # Deny
    subjects = list(np.linspace(33, 54, 22, dtype=int))
    
    
    for subject in subjects:
        for fold in range(10):
            framework = Main()
            framework.fit(subject=subject, fold=fold)
