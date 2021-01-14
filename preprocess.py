import pickle
import json
import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
import numpy as np

curr_dir = os.getcwd()
pickles = "/cfg/data.pickle"

def preprocess(data):

    words = []
    docs_x = []
    docs_y = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]

    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
        #wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open(curr_dir + pickles, "wb") as f:
        pickle.dump((words, labels, training, output), f)