import pickle
import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from utils import FIRST, FOURTH, SECOND, THIRD, codes, get_config
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

#converts from numpy array to list
def np_sentence_to_list(L_sent):
    newsent = []
    for sentance in L_sent:
        temp = []
        for word in sentance:
            temp.append(word.tolist())
        newsent.append(temp)
    return newsent


def train_epoch(config, model, X, y, loss_fn, optimizer, epoch):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    #random.shuffle(data)
    for i in range(len(X)):
        #sent, label = data[i]
        sent = Variable(torch.Tensor(np_sentence_to_list(X[i])))
        label = Variable(torch.LongTensor(y[i]))

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

        optimizer.zero_grad()
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
    return avg_loss, acc


def evaluate(config, model, X, y):
    model.eval()
    truth_res = []
    pred_res = []
    test_data = list(zip(X, y))
    random.shuffle(test_data)
    X, y = zip(*train_data)
    for i in range(len(X)):
        #sent, label = data[i]
        sent = Variable(torch.Tensor(np_sentence_to_list(X[i])))
        label = Variable(torch.LongTensor(y[i]))

        truth_res.append(label.data[0])
        model.hidden = model.init_hidden()

        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()[0]
        pred_res.append(pred_label)

    acc = get_accuracy(truth_res, pred_res)
    return acc

#if doing batched row in data is actually a list of all the sentences of a particular length, largest list is of length 39000,
def train(config, data, code):
    #random.shuffle(data)
    num_tr = int(len(data) * 0.8)
    X = []
    Y = []
    for row in data:
        random.shuffle(row)
        #print(row)
        X.append([innerrow[0] for innerrow in row])
        Y.append([innerrow[1] for innerrow in row])
    #X = [row[0] for row in data]
    #y = [row[1] for row in data]

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in xrange(len(X)):
          X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X[i], Y[i], test_size=0.2, random_state=42)
          X_train.append(X_train_temp)
          X_test.append(X_test_temp)
          y_train.append(y_train_temp)
          y_test.append(y_test_temp)

    #train_data = data[:num_tr]
    #test_data = data[num_tr:]

    #print('{} training samples, {} testing samples'.format(
        #len(train_data), len(test_data)))

    label_size = 2

    EMBEDDING_DIM = config.feature_size
    HIDDEN_DIM = 128
    EPOCH = 50
    best_acc = 0.0
    model = LSTMClassifier(
        config,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        label_size=label_size)

    # loss_function = nn.NLLLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    train_accs = []
    test_accs = []

    for i in range(EPOCH):
        train_data = list(zip(X_train, y_train))
        random.shuffle(train_data)
        X_train, y_train = zip(*train_data)

        train_loss, train_acc = train_epoch(config, model, X_train, y_train, loss_fn,
                                            optimizer, i)
        losses.append(train_loss)
        train_accs.append(train_acc)

        acc = evaluate(config, model, X_test, y_test)
        test_accs.append(acc)
        print("epoch #: {}".format(i))
        print('Test Acc: {:.2f}%'.format(acc * 100))
        print('')

        if acc >= best_acc:
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
    embedding_data = word2vec(config, code=code)
    train(config, embedding_data, code)
