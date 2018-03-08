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
