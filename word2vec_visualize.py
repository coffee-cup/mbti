# coding: utf-8

import random

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

from utils import get_config


def load_model(config):
    """Load the word2vec model from disk."""
    return Word2Vec.load(config.embeddings_model)


def get_vocab(model, n):
    """Returns n labels and vectors belonging to word2vec model."""
    labels = []
    tokens = []

    i = 0

    items = list(model.wv.vocab.items())
    random.shuffle(items)
    for word, _ in items:
        tokens.append(model[word])
        labels.append(word)

        i += 1
        if i >= n:
            break

    return labels, tokens


def plot3d(labels, tokens):
    """Plot word vectors in 3d."""
    print('Plotting {} points in 3d'.format(len(labels)))

    # Reduce dimensionality with TSNE
    tsne_model = TSNE(
        perplexity=40,
        n_components=3,
        init='pca',
        n_iter=2500,
        learning_rate=600)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    z = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='.', edgecolors='none')

    for i, l in enumerate(labels):
        ax.text(x[i], y[i], z[i], l)

    plt.show()


def plot2d(labels, tokens):
    """Plot word vectors in 2d."""
    print('Plotting {} points in 2d'.format(len(labels)))

    # Reduce dimensionality with TSNE
    tsne_model = TSNE(
        perplexity=40,
        n_components=2,
        init='pca',
        n_iter=2000,
        learning_rate=500)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(x)):
        ax.scatter(x[i], y[i], marker='.', c='b', s=100, edgecolors='none')
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.show()


if __name__ == '__main__':
    config = get_config()
    model = load_model(config)

    N = 100
    labels, tokens = get_vocab(model, N)
    plot2d(labels, tokens)
