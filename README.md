# MBTI Personality Type Predictor

This project is a work in progress data mining project for UVic Seng 474 - Data Mining.

## Dataset

Download and extract the csv file from [Kaggle](https://www.kaggle.com/datasnaek/mbti-type/version/1).

Place the extracted file in a directory called `data`.

## Usage

You can run all the steps at once with

```sh
python main.py
```

Or run each step individually

```sh
python preprocess.py
python word2vec.py
```

To see all of the config variables, run

```sh
python main --help
```

```
usage: main.py [-h] [--data_dir DATA_DIR] [--raw_csv_file RAW_CSV_FILE]
               [--pre_save_file PRE_SAVE_FILE]
               [--force_preprocessing FORCE_PREPROCESSING]
               [--embeddings_model EMBEDDINGS_MODEL]
               [--embeddings_file EMBEDDINGS_FILE] [--num_threads NUM_THREADS]
               [--feature_size FEATURE_SIZE] [--min_words MIN_WORDS]
               [--distance_between_words DISTANCE_BETWEEN_WORDS]
               [--epochs EPOCHS] [--force_word2vec FORCE_WORD2VEC]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory to save/read all data files from

Preprocessing:
  --raw_csv_file RAW_CSV_FILE
                        Filename of csv file downloaded from Kaggle
  --pre_save_file PRE_SAVE_FILE
                        Filename to save preprocessed csv file as
  --force_preprocessing FORCE_PREPROCESSING
                        Whether or not to do preprocessing even if output csv
                        file is found

Word2Vec:
  --embeddings_model EMBEDDINGS_MODEL
                        Filename to save word2vec model to
  --embeddings_file EMBEDDINGS_FILE
                        Filename to save mbti data with word vectors to
  --num_threads NUM_THREADS
                        Number of threads to use for training word2vec
  --feature_size FEATURE_SIZE
                        Number of features to use for word2vec
  --min_words MIN_WORDS
                        Minimum number of words for word2vec
  --distance_between_words DISTANCE_BETWEEN_WORDS
                        Distance between words for word2vec
  --epochs EPOCHS       Number of epochs to train word2vec for
  --force_word2vec FORCE_WORD2VEC
                        Whether or not to create word embeddings even if
                        output word2vec file is found
```

# Data Format

The format of the data can get a little confusing. Hopefully this clears things up.

For the following, `N = number of rows (samples) we have**.

_Note: All filepaths are prefixed with the `data` directory._

## Preprocessing

### Input

Raw CSV file coming from Kaggle. The location of the input file is given by `config.raw_csv_file`.

```python
preprocess(config) # nothing returned, new csv file saved
```

### Output

The file is preprocessed by splitting each row into a new row for each individual post. Stopwords, numbers, links, and punctuation are removed and the text is set to all lowercase. The file is saved to `config.pre_save_file`.

## Word2Vec

This is the data that will be mainly used for training/testing. Multiple output types can be specified depending if are training to classify all 16 classes, or doing a binary classification for each of the 4 character codes.

### Input

Preprocessed CSV file. The location of the input file is given by `config.pre_save_file`.

As input you also need to give the personality "character code". The options are imported from `utils.py`.

```python
from utils import FIRST, FOURTH, SECOND, THIRD
embedding_data = word2vec(config, code=ALL) # Defaults to ALL
```

### Output

The output will all be numbers, no strings will be present. 

For each row, the first element will be the label vector and the second element will be the sentence data.

#### Sentence

The sentence data is a list of words vectors, so may be a different length for each row.

#### Word Vector

Each word vector is a vector of length 300.

#### Label

The label depends on the `code` option specified.

**ALL***

```
row = [label, sentence]
```

The label will be a length 16 vector which is one-hot encoded. You can use the `utils.one_hot_to_type` function to convert from a one-hot encoding to a personality type.

_For example_

```python
# Get one hot encoding
Y = one_hot_encode_type('INTJ')
print(Y)
# => [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# Get personality type
t = one_hot_to_type(Y)
print(t)
# => INTJ

```

**FIRST, SECOND, THIRD, FOURTH**

The label will be a length 1 vector which is either 0 or 1. When training, the output should be just a binary classification. To get what the character was based on the binary classification, you can use the `utils.get_char_for_binary` function.

_For example_

```python
# Consider the third character (T or F)
code = THIRD

# Get binary class
b = get_binary_for_code(code, 'ESTP')
print(b)
# => 0

# Get character for class
c = get_char_for_binary(code, b)
print(c)
# => T
```
