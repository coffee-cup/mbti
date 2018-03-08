import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from utils import *
from sklearn.metrics import classification_report
from word2vec import word2vec
from sklearn.cross_validation import train_test_split


def meanEmbeddingTransform(feature):
        return np.array([[np.mean([np.mean(sentence) for sentence in sentences if np.any(sentence)==True] or [0] )] for sentences in feature])

# with open("./data/preprocessed.csv") as labelFile:

config = get_config()
data = word2vec(config)
# w2v = {line[0]: line[1:] for line in data}

# print w2v.keys
data_train, data_test = train_test_split(data, test_size = 0.2)
# print data_train
y_train, X_train = map(list, zip(*data_train))
y_test, X_test = map(list, zip(*data_test)) 

y_train = np.array(y_train)


y_test = np.array(y_test)

X_train = meanEmbeddingTransform(X_train)
X_test = meanEmbeddingTransform(X_test)

clf = GaussianNB()
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print classification_report(y_test, prediction)