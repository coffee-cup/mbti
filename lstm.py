import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import FIRST, FOURTH, SECOND, THIRD, get_config
from word2vec import word2vec

random.seed(1)
torch.manual_seed(1)


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
        x = embeds.view(embeds.size(0), 1, -1)
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


def train_epoch(config, model, data, loss_fn, optimizer, epoch):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in data:
        sent = Variable(torch.Tensor(sent))
        label = Variable(torch.LongTensor(label))

        truth_res.append(label.data[0])
        model.hidden = model.init_hidden()

        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()[0]
        pred_res.append(pred_label)

        # print('Pred: {} Actual: {}'.format(pred_label, label.data[0]))
        # print(pred.data[0].numpy())
        # print('')

        # pred = pred.view([model.label_size])
        # print(pred)

        model.zero_grad()
        loss = loss_fn(pred, label)
        avg_loss += loss.data[0]
        count += 1

        if count % 100 == 0:
            print('\tEpoch: {} Iteration: {} Loss: {}'.format(
                epoch, count, loss.data[0]))

        loss.backward()
        optimizer.step()

    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print('Epoch: {} Avg Loss: {} Acc: {:.2f}%'.format(epoch, avg_loss,
                                                       acc * 100))


def evaluate(model, data):
    model.eval()
    truth_res = []
    pred_res = []

    for sent, label in data:
        sent = Variable(torch.Tensor(sent))
        label = Variable(torch.LongTensor(label))

        truth_res.append(label.data[0])
        model.hidden = model.init_hidden()

        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()[0]
        pred_res.append(pred_label)

    acc = get_accuracy(truth_res, pred_res)
    return acc


def train(config, data):
    random.shuffle(data)
    num_tr = int(len(data) * 0.8)

    X = [row[0] for row in data]
    y = [row[1] for row in data]

    train_data = data[:num_tr]
    test_data = data[num_tr:]

    print('{} training samples, {} testing samples'.format(
        len(train_data), len(test_data)))

    label_size = 2

    EMBEDDING_DIM = config.feature_size
    HIDDEN_DIM = 50
    EPOCH = 100
    best_acc = 0.0
    model = LSTMClassifier(
        config,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        label_size=label_size)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(EPOCH):
        random.shuffle(train_data)
        print('Epoch: {}'.format(i))
        train_epoch(config, model, train_data, loss_function, optimizer, i)

        acc = evaluate(model, test_data)
        print('Test Acc: {:.2f}%'.format(acc * 100))
        print('')


if __name__ == '__main__':
    config = get_config()

    embedding_data = word2vec(config, code=THIRD)
    train(config, embedding_data)
