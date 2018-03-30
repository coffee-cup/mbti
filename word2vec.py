import csv
import os
import random
import re
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, word2vec

from tqdm import trange
from utils import (ALL, FIRST, FOURTH, SECOND, THIRD, get_binary_for_code,
                   get_config, one_hot_encode_type)


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
        sorted_vocab=1,
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


def convert_posts_to_vectors(config, data, model):
    """Convert the text post to a vector

    :data df.read_csv(pre_processed_csv).values
    :model word2vec model

    Returns array of length N of format [mbti type, sentence array]
        sentence array is an array of words where each word is a 300 dim vector
    """
    print('Converting text data to vectors...')

    N = len(data)
    # Get number of samples to use from config if not -1
    if config.num_samples != -1:
        N = config.num_samples

    # Shuffle data
    random.shuffle(data)
    embedded_data = []
    for idx in trange(N):
        row = data[idx]
        post = row[1]
        mbti_type = row[2]

        sentence = []
        for word in post.split(' '):
            if word in model.wv.index2word:
                vec = model.wv[word]
                sentence.append(vec)
        if len(sentence) > 0:
            embedded_data.append([mbti_type, sentence])

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


def get_code_data(code, embedding_data):
    """Get data with label as binary specifying a specific personality type code."""
    newdata = []
    for row in embedding_data:
        c = get_binary_for_code(code, row[0])
        newdata.append([row[1], [c]])
    return newdata


def get_one_hot_data(embedding_data):
    """Get data with label one-hot encoded for all possible classes."""
    newdata = []
    for row in embedding_data:
        Y = one_hot_encode_type(row[0])
        newdata.append([row[1], Y])
    return newdata


def word2vec(config, code=ALL):
    """Create word2vec embeddings

    :config user configuration
    """
    print('\n--- Creating word embeddings')

    embedding_data = None
    pre_data = pd.read_csv(config.pre_save_file).values
    if os.path.isfile(config.embeddings_model) and not config.force_word2vec:
        # Load model from file
        model = load_word2vec(config.embeddings_model)
    else:
        # Train model
        words = extract_words(pre_data)
        model = create_word2vec_model(words, config)

        # Save model to disk
        model.save(config.embeddings_model)

    # Create data with labels and word embeddings
    embedding_data = convert_posts_to_vectors(config, pre_data, model)

    if code == ALL:

        embedding_data = get_one_hot_data(embedding_data)
        return batch_embeddings(embedding_data)
    return batch_embeddings(get_code_data(code, embedding_data))


def average_sentence_length(embedding_data):
    """Returns the average number of words in a sentence"""
    total_words = 0
    for row in embedding_data:
        total_words += float(len(row[1]))
    return total_words / len(embedding_data)

def batch_embeddings(embeddings):
    batched_embeddings = []
    lengths = set()
    for row in embeddings:
        lengths.add(len(row[0]))
    max_length = max(lengths)
    #min_length = len(min(l[0] for l in embeddings))
    #print(max_length)
    for i in xrange(max_length+1):
        temp = []
        for row in embeddings:

            if(len(row[0]) == i):
                temp.append(row)
        if(len(temp) != 0):
            batched_embeddings.append(temp)
            #print(len(temp))

    return batched_embeddings
# SHIT DON'T WORK
# def get_embeddings(config):
#     """Returns data rows with embeddings from disk."""
#     data = pd.HDFStore(config.embeddings_file)
#     return data['data'].values

# def save_embeddings(config, embedding_data):
#     """Saves data rows with embeddings to disk."""
#     x = pd.HDFStore(config.embeddings_file)
#     x.append('data', pd.DataFrame(embedding_data))
#     x.close()

if __name__ == "__main__":
    config = get_config()

    embedding_data = word2vec(config)
    batched_embeddings = batch_embeddings(embedding_data)
    print(len(embedding_data), len(batched_embeddings))
    print('Created word embeddings')

    print('Rows: {}'.format(len(embedding_data)))
    print('Average number of words: {}'.format(
        average_sentence_length(embedding_data)))
