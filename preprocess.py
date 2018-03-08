import os.path
import re
import string

import numpy as np
import pandas as pd

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


if __name__ == '__main__':
    config = get_config()
    posts = preprocess(config)

    # Visualize the preprocessing
    print('Preprocess Complete!')
    print('Here are the first 2 rows')
    print(posts[0:2])
