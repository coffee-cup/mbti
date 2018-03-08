# Utils file for useful functions

import os
import pickle

from config import parse_config, print_usage


def get_config():
    """Gets config and creates data_dir."""
    config, unparsed = parse_config()

    # If we have unparsed args, print usage and exit
    if len(unparsed) > 0:
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

    return config


def save_data(data, filename):
    """Pickle and save some data to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_data(filename):
    """Reads a pickle file from disk and returns the data."""
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
