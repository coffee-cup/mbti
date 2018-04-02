import pickle
import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from utils import FIRST, FOURTH, SECOND, THIRD, codes, get_config
from word2vec import word2vec

# random.seed(1)
# torch.manual_seed(1)


class MbtiDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, config, embedding_dim, hidden_dim, label_size):
        print('label size {}'.format(label_size))
        super(LSTMClassifier, self).__init__()
        self.config = config
        self.label_size = label_size
        self.hidden_dim = hidden_dim
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h1 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        h2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))

        if self.config.use_cuda:
            h1 = h1
            h2 = h2
        return (h1, h2)

    def forward(self, embeds):
        x = embeds
        # print('embeds size {}'.format(embeds.size()))
        # x = embeds.view(embeds.size(0), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch(model, dataloader, loss_fn, optimizer, epoch):
    '''Train a single epoch.'''
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []

    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched
        inputs = Variable(torch.stack(inputs))
        labels = Variable(torch.stack(labels)).view(-1)

        truth_res.append(labels.data[0])
        model.hidden = model.init_hidden()

        pred = model(inputs)
        pred_label = pred.data.max(1)[1].numpy()[0]
        pred_res.append(pred_label)

        optimizer.zero_grad()
        loss = loss_fn(pred, labels)
        avg_loss += loss.data[0]
        count += 1

        if count % 100 == 0:
            print('\tBatch: {} Iteration: {} Loss: {}'.format(
                i_batch, count, loss.data[0]))

        loss.backward()
        optimizer.step()

    avg_loss /= count
    acc = get_accuracy(truth_res, pred_res)
    print('Epoch: {} Avg Loss: {} Acc: {:.2f}%'.format(epoch, avg_loss,
                                                       acc * 100))
    return avg_loss, acc


def evaluate(model, dataloader):
    model.eval()

    truth_res = []
    pred_res = []

    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched
        inputs = Variable(torch.stack(inputs))
        labels = Variable(torch.stack(labels)).view(-1)

        truth_res.append(labels.data[0])
        model.hidden = model.init_hidden()

        pred = model(inputs)
        pred_label = pred.data.max(1)[1].numpy()[0]
        pred_res.append(pred_label)

    acc = get_accuracy(truth_res, pred_res)
    return acc


def lstm(config, embedding_data, code):
    X = [row[0] for row in embedding_data]
    y = [row[1] for row in embedding_data]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_dataset = MbtiDataset(X_train, y_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4)

    test_dataset = MbtiDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4)

    label_size = 2
    EMBEDDING_DIM = config.feature_size
    HIDDEN_DIM = 128
    EPOCH = 10
    best_acc = 0.0

    model = LSTMClassifier(
        config,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        label_size=label_size)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=1e-4)

    losses = []
    train_accs = []
    test_accs = []

    for i in range(EPOCH):
        avg_loss = 0.0

        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn,
                                            optimizer, i)
        losses.append(train_loss)
        train_accs.append(train_acc)

        acc = evaluate(model, test_dataloader)
        test_accs.append(acc)

        print('Epoch #{} Test Acc: {:.2f}%'.format(i, acc * 100))
        print('')

        if acc > best_acc:
            best_acc = acc

    save_data = {
        'best_acc': best_acc,
        'losses': losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'personality_char': code + 1,
        'letters': codes[code]
    }

    print('Best Acc: {:.2f}%'.format(best_acc * 100))

    with open('lstm_save', 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    config = get_config()

    code = FIRST
    embedding_data = word2vec(config, code=code, batch=False)
    lstm(config, embedding_data, code)
