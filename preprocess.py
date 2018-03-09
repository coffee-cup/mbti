import os.path
import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import *

# Regular expression to match punctuation
reg_punc = re.compile('[%s]' % re.escape(string.punctuation))

# Regular expression to match links
reg_link = re.compile('http\S+', flags=re.MULTILINE)

# Regular expression to match non-characters
reg_alpha = re.compile('[^a-zA-Z ]')

# Regular expression to match all whitespace
reg_spaces = re.compile('\s+', flags=re.MULTILINE)


def filter_text(post):
    """Decide whether or not we want to use the post."""
    return len(post) > 0


def preprocess_text(post):
    """Remove any junk we don't want to use in the post."""

    # Remove links
    post = reg_link.sub('', post)

    # All lowercase
    post = post.lower()

    # Remove non-alpha chars
    post = reg_alpha.sub('', post)

    # Replace multiple whitespace with single space
    post = reg_spaces.sub(' ', post)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(post)
    post = [w for w in word_tokens if not w in stop_words]
    post = ' '.join(post)

    # Strip whitespace
    posts = post.strip()

    return post


def create_new_rows(row):
    """Create new rows of the data by preprocessing the individual posts and filtering out bad ones."""
    posts = row['posts'].split('|||')
    rows = []

    for p in posts:
        p = preprocess_text(p)
        if not filter_text(p):
            continue
        rows.append({'type': row['type'], 'post': p})
    return rows


def preprocess(config):
    """Preprocess the data using the config.

    :config user configuration
    """
    print('\n--- Preprocessing')

    if os.path.isfile(config.pre_save_file) and not config.force_preprocessing:
        df = pd.read_csv(config.pre_save_file)
        return df.values

    df = pd.read_csv(config.raw_csv_file)
    newrows = []
    for index, row in df.iterrows():
        newrows += create_new_rows(row)

    df = pd.DataFrame(newrows)
    df.to_csv(config.pre_save_file)

    return df.values


def get_count(posts, fn):
    counts_dict = {}
    for row in posts:
        label = row[-1]
        l = fn(label)
        if counts_dict.get(l) is None:
            counts_dict[l] = 1
        else:
            counts_dict[l] += 1

    counts = []
    counts = map(lambda x: (x[0], x[1]), counts_dict.items())
    return sorted(counts, key=lambda t: -t[1])


def print_counts(counts):
    for t in counts:
        print('{} {}'.format(t[0], t[1]))
    print('')


if __name__ == '__main__':
    config = get_config()
    posts = preprocess(config)

    # Visualize the preprocessing
    print('Preprocess Complete!')
    print('{} Total rows'.format(len(posts)))
    print('Here are the first 2 rows')
    print(posts[0:2])

    print('\nCount of labels for all 16 classes and each character\n')

    print('All')
    print_counts(get_count(posts, lambda x: x))

    print('First')
    print_counts(get_count(posts, lambda x: x[0]))

    print('Second')
    print_counts(get_count(posts, lambda x: x[1]))

    print('Third')
    print_counts(get_count(posts, lambda x: x[2]))

    print('Fourth')
    print_counts(get_count(posts, lambda x: x[3]))
