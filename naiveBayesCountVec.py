import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

X_data = []
y_data = []

def regex_tokenizer(doc):
    """Return a function that split a string in sequence of tokens"""
    return doc.split(' ')

with open("./data/preprocessed.csv") as csvFile:
    reader = csv.reader(csvFile)
    x = 0
    for line in reader:
        if line[1] ==" ":
            continue
        X_data.append(line[1])
        y_data.append(line[2])
        x +=1;
        if x == 20000:
            break         

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)

vectorizer = CountVectorizer(lowercase=False, stop_words=None,  max_df=1.0, min_df=1, max_features=None, tokenizer=regex_tokenizer )

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print X_train

y_test = np.asarray(y_test)
y_train = np.asarray(y_train)

print len(X_train)
print len(y_train)

clf = MultinomialNB()
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
print classification_report(y_test, prediction)