import pickle
import json
import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
import numpy as np

#Necessary paths
curr_dir = os.getcwd()
pickles = "/cfg/data.pickle"

def preprocess(data):

    #Temporary lists needed for preprocessing the knowledge base data.
    #The words list is used in intent_predictor.py in order to construct a vector of the user input
    words = []
    docs_x = []
    docs_y = []
    labels = []

    #Fills the docs_x and words lists with all tokenized questions/patterns and the docs_y file with the corresponding labels.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp_words = nltk.word_tokenize(pattern)
            words.extend(temp_words)
            docs_x.append(temp_words)
            docs_y.append(intent["tag"])
        #Appends the intent tags in the labels list. This is needed for the intent_predictor.py function, in order to find the most likely label/intent of the user.
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #Every word is lemmatized/stemmed. Helps keep the knowledge base smaller.
    #words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]

    #Sorting isn't necessary, but it helps when playing with the code.
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #The two lists used in the nn_model_handler.py. They are used to fit our neural network model with our now processed data.
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        temp_words = [lemmatizer.lemmatize(w.lower()) for w in doc]
        #temp_words = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in temp_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    #Save data to a pickles dump
    with open(curr_dir + pickles, "wb") as f:
        pickle.dump((words, labels, training, output), f)