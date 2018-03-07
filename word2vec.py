import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv, re
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np

def create_word2vec_model(posts, num_threads, feature_size, min_words, distance_between_words, epochs):

        print("Training model")
        model = word2vec.Word2Vec(posts, workers=num_threads, size=feature_size, min_count=min_words, window=distance_between_words, sample=1e-3, iter=epochs)
        model.init_sims(replace=True)
        model.save("temp_model")

        return model
        
#does some more preprocessing of the data
def extract_data(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        posts = []
        #takes out all numbers and empty posts
        for row in reader:
            post = row.get('post')

            if(len(post.strip()) > 0):
                text = re.sub("[^a-zA-Z ]","", post)  
                words = text.split()
                posts.append(words)
                
        return posts

#matches each word of a post to the index in the model. Note not every word is in the model       
def convert_data_to_index(posts, model):
    seen = set()
    index_data = []

    for post in posts:
        index_of_post = []
        for word in post:
            
            if word in model.wv and word not in seen:
                seen.add(word)
                index_of_post.append(model.wv.vocab[word].index)

        index_data.append(index_of_post)
            
    return index_data   

#Converts the keyedVectors of the model into numpy arrays this way can use it in pytorch or whatever        
def get_embeddings(model):
    
    num_features = len(model[model.wv.vocab.keys()[0]])
        
    embedded_weights = np.zeros((len(model.wv.vocab), num_features))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedded_weights[i] = embedding_vector
    
    return embedded_weights
    
if __name__ == "__main__":
    posts = extract_data('mbti_preprocessed.csv')
    #train the model the first time, takes a couple mins so after do this once just want to load it
    model = create_word2vec_model(posts, 4, 300, 10, 10, 10) #comment out this line after ran once
   
    #model = Word2Vec.load("temp_model") #uncomment this line if running again
    weights = get_embeddings(model)
    indexed = convert_data_to_index(posts, model)

