# The main file to run all steps

from preprocess import preprocess
from utils import (ALL, FIRST, FOURTH, SECOND, THIRD, get_char_for_binary,
                   get_config, one_hot_to_type)
from word2vec import word2vec

if __name__ == '__main__':
    config = get_config()

    preprocess(config)

    # 16 classes
    embedding_data = word2vec(config, code=ALL)
    example = embedding_data[10]
    print('Label is {}'.format(one_hot_to_type(example[0])))

    # Binary class (third class)
    # code = FOURTH
    # embedding_data = word2vec(config, code=code)
    # example = embedding_data[10]
    # print('Binary label is {}'.format(get_char_for_binary(code, example[0])))
