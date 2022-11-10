from tqdm import tqdm
import torch
from torch import nn, optim

from model import Model

class Train:
    def __init__(self, arg):
        self.arg = arg

        self.gtnModel = Model(args=arg).to(self.arg.device)
        self.optim = optim.Adam(self.gtnModel.parameters(), lr=self.arg.lrate)
        self.criterions = nn.CrossEntropyLoss(label_smoothing=0.12).to(self.arg.device)

        self.best_performance = .0

    def fit(self, train, val):
        for iter in range(self.arg.iteration):
            batch_losses, batch_accuracy = 0, 0
            self.gtnModel.train()
            for x, y, a in tqdm(train):
                x = x.to(self.arg.device)
                y = y.to(self.arg.device)
                a = a.to(self.arg.device)

                self.optim.zero_grad()
                y_hat, att = self.gtnModel(x, a)
                losses = self.criterions(y_hat, y.squeeze())
                losses.backward()

                torch.nn.utils.clip_grad_norm_(self.gtnModel.parameters(), 1)
                self.optim.step()

                batch_losses += losses.item()
            train_losses  = batch_losses / len(train)
            test_loss, test_acc = self.testing(val)

            if self.best_performance < test_acc:
                self.best_performance = test_acc
                self.save_parameter()

            print(f'Iteration: {iter} | Train Loss: {train_losses: .3f} | Test Loss: {test_loss: .3f} | Test Acc.: {test_acc: .3f} | Best. {self.best_performance: .3f}\n')


    def testing(self, val):
        batch_losses, batch_accuracy = .0, .0
        self.gtnModel.train()
        for x, y, a in tqdm(val):
            x = x.to(self.arg.device)
            y = y.to(self.arg.device)
            a = a.to(self.arg.device)

            self.optim.zero_grad()
            y_hat, att = self.gtnModel(x, a)
            losses = self.criterions(y_hat, y.squeeze())
            losses.backward()

            torch.nn.utils.clip_grad_norm_(self.gtnModel.parameters(), 1)
            self.optim.step()

            batch_losses += losses.item()
            batch_accuracy += self.metrices(y, y_hat)

        test_loss = batch_losses / len(val)
        test_accuracy = batch_accuracy / self.arg.test_len
        return test_loss, test_accuracy

    def metrices(self, y, y_hat):
        accuracy = 0
        y_hat = torch.argmax(y_hat, dim=1)
        for t in range(y.shape[0]):
            if y[t] == y_hat[t]:
                accuracy += 1
        return accuracy

    def save_parameter(self):
        params = {'model': self.gtnModel.state_dict(),
                  'optim': self.optim.state_dict()}
        torch.save(params, f'weights/S{self.arg.subject_id: 02d}_F{self.arg.k_fold: 02d}.pth')


