# Utils file for useful functions

import os

import numpy as np

from config import parse_config, print_usage

first_codes = ['I', 'E']
second_codes = ['S', 'N']
third_codes = ['T', 'F']
fourth_codes = ['J', 'P']
codes = [first_codes, second_codes, third_codes, fourth_codes]

FIRST = 0
SECOND = 1
THIRD = 2
FOURTH = 3
ALL = 16

personality_types = [
    'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP',
    'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
]


def one_hot_encode_type(t):
    i = personality_types.index(t)
    Y = np.zeros(len(personality_types))
    Y[i] = 1
    return Y.astype(int).tolist()


def one_hot_to_type(Y):
    i = np.where(Y == 1)[0][0]
    return personality_types[i]


def get_binary_for_code(code, personality_type):
    c = codes[code]
    return int(personality_type[code] != c[0])


def get_char_for_binary(code, binary):
    if type(binary) is list:
        binary = binary[0]
    c = codes[code]
    return c[binary]


def get_config(return_unparsed=False):
    """Gets config and creates data_dir."""
    config, unparsed = parse_config()

    # If we have unparsed args, print usage and exit
    if len(unparsed) > 0 and not return_unparsed:
        print_usage()
        exit(1)

    def append_data_dir(p):
        return os.path.join(config.data_dir, p)

    # Append data_dir to all filepaths
    config.pre_save_file = append_data_dir(config.pre_save_file)
    config.raw_csv_file = append_data_dir(config.raw_csv_file)
    config.embeddings_model = append_data_dir(config.embeddings_model)
    config.embeddings_file = append_data_dir(config.embeddings_file)

    # Create data_dir if it doesn't exist
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    if return_unparsed:
        return config, unparsed

    return config
