import os.path
import re
import string

import numpy as np
import pandas as pd

from config import get_config, print_usage


def filter_text(post):
    """Decide whether or not we want to use the post."""
    return len(post) > 0


reg_punc = re.compile('[%s]' % re.escape(string.punctuation))


def preprocess_text(post):
    """Remove any junk we don't want to use in the post."""

    # Remove links
    post = re.sub(r'http\S+', '', post, flags=re.MULTILINE)

    # All lowercase
    post = post.lower()

    # Remove puncutation
    post = reg_punc.sub('', post)

    return post


def create_new_rows(row):
    posts = row['posts'].split('|||')
    rows = []

    for p in posts:
        p = preprocess_text(p)
        if not filter_text(p):
            continue
        rows.append({'type': row['type'], 'post': p})
    return rows


def preprocess(config):
    """Preprocess the data using the config."""
    print('\n--- Processing')

    if os.path.isfile(config.pre_save_file) and not config.force_preprocessing:
        df = pd.read_csv(config.pre_save_file)
        return df.values

    df = pd.read_csv(config.raw_csv_file)
    newrows = []
    for index, row in df.iterrows():
        newrows += create_new_rows(row)

    df = pd.DataFrame(newrows)
    unique = df.groupby('type').nunique()

    df.to_csv(config.pre_save_file)

    return df.values


if __name__ == '__main__':
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    posts = preprocess(config)

    # Visualize the preprocessing
    print('Preprocess Complete!')
    print('Here are the first 2 rows')
    print(posts[0:2])
