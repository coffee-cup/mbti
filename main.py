# The main file to run all steps

from preprocess import preprocess
from utils import *
from word2vec import word2vec

if __name__ == '__main__':
    config = get_config()

    preprocess(config)
    embedding_data = word2vec(config)
