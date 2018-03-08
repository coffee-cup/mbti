import argparse

# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='Directory to save/read all data files from')

# Arguments for preprocessing
preprocessing_arg = add_argument_group('Preprocessing')

preprocessing_arg.add_argument(
    '--raw_csv_file',
    type=str,
    default='mbti_1.csv',
    help='Filename of csv file downloaded from Kaggle')

preprocessing_arg.add_argument(
    '--pre_save_file',
    type=str,
    default='mbti_preprocessed.csv',
    help='Filename to save preprocessed csv file as')

preprocessing_arg.add_argument(
    '--force_preprocessing',
    type=str2bool,
    default=False,
    help='Whether or not to do preprocessing even if output csv file is found')

# Arguments for word2vec
word2vec_arg = add_argument_group('Word2Vec')

word2vec_arg.add_argument(
    '--embeddings_model',
    type=str,
    default='embeddings_model',
    help='Filename to save word2vec model to')

word2vec_arg.add_argument(
    '--embeddings_file',
    type=str,
    default='vector_data',
    help='Filename to save mbti data with word vectors to')

word2vec_arg.add_argument(
    '--num_threads',
    type=int,
    default=4,
    help='Number of threads to use for training word2vec')

word2vec_arg.add_argument(
    '--feature_size',
    type=int,
    default=300,
    help='Number of features to use for word2vec')

word2vec_arg.add_argument(
    '--min_words',
    type=int,
    default=10,
    help='Minimum number of words for word2vec')

word2vec_arg.add_argument(
    '--distance_between_words',
    type=int,
    default=10,
    help='Distance between words for word2vec')

word2vec_arg.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='Number of epochs to train word2vec for')

word2vec_arg.add_argument(
    '--force_word2vec',
    type=str2bool,
    default=False,
    help=
    'Whether or not to create word embeddings even if output word2vec file is found'
)

word2vec_arg.add_argument(
    '--num_samples',
    type=int,
    default=-1,
    help='Number of samples to return from word2vec. -1 for all samples')


def parse_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()
