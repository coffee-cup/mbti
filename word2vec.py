import csv
import os
import re

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, word2vec

from tqdm import trange
from utils import *


def create_word2vec_model(data, config):
    """Train a word2vec model

    :data Data to train on
    :num_threads Number of threads to use while training
    :feature_size Number of features to use
    :min_words Ignore all words with total frequency lower than this
    :distance_between_words Maximum distance between the
        current and predicted word within a sentence
    :epochs Number of iterations to run over the corpus
    """
    print('Training model...')
    model = Word2Vec(
        data,
        workers=config.num_threads,
        size=config.feature_size,
        min_count=config.min_words,
        window=config.distance_between_words,
        sample=1e-3,
        iter=config.epochs)
    model.init_sims(replace=True)

    return model


def load_word2vec(name):
    """Load word2vec model from file."""
    return Word2Vec.load(name)


def extract_words(data):
    """Extract words from posts.

    :data preprocessed data
    """
    words = []
    for row in data:
        post = row[1]
        words.append(post.split())

    return words


def convert_data_to_index(posts, model):
    """Match each word of a post to the index in the model.
    Note: not every word is in the model
    """
    index_data = []

    for post in posts:
        index_of_post = []
        for word in post:

            if word in model.wv:
                index_of_post.append(model.wv.vocab[word].index)

        index_data.append(index_of_post)

    return index_data


def convert_posts_to_vectors(data, model):
    """Convert the text post to a vector

    :data df.read_csv(pre_processed_csv).values
    :model word2vec model

    Returns array of length N of format [mbti type, sentence array]
        sentence array is an array of words where each word is a 300 dim vector
    """
    print('Converting text data to vectors...')

    embedded_data = []
    for idx in trange(len(data)):
        row = data[idx]
        post = row[1]
        mbti_type = row[2]

        sentence = []
        for word in post.split(' '):
            if word in model.wv.index2word:
                vec = model.wv[word]
                sentence.append(vec)
        embedded_data.append([mbti_type, sentence])

        # TODO: Remove this after finding better way to save data
        if idx >= 2000:
            break

    return embedded_data


def get_embeddings(model):
    """Convert the keyedVectors of the model into numpy arrays."""
    num_features = len(model[list(model.wv.vocab.keys())[0]])

    embedded_weights = np.zeros((len(model.wv.vocab), num_features))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedded_weights[i] = embedding_vector

    return embedded_weights


def word2vec(config):
    """Create word2vec embeddings

    :config user configuration
    """
    print('\n--- Creating word embeddings')

    embedding_data = None
    if os.path.isfile(config.embeddings_file) and not config.force_word2vec:
        # Load data with word vectors from file
        embedding_data = read_data(config.embeddings_file)
    else:
        # Train model
        pre_data = pd.read_csv(config.pre_save_file).values
        words = extract_words(pre_data)
        model = create_word2vec_model(words, config)

        # Save model to disk
        model.save(config.embeddings_model)

        # Create data with labels and word embeddings
        embedding_data = convert_posts_to_vectors(pre_data, model)

        # Save to disk
        save_data(embedding_data, config.embeddings_file)

    return embedding_data


def average_sentence_length(embedding_data):
    """Returns the average number of words in a sentence"""
    total_words = 0
    for row in embedding_data:
        total_words += float(len(row[1]))
    return total_words / len(embedding_data)


if __name__ == "__main__":
    config = get_config()

    embedding_data = word2vec(config)

    print('Created word embeddings')

    print('Rows: {}'.format(len(embedding_data)))
    print('Average number of words: {}'.format(
        average_sentence_length(embedding_data)))
