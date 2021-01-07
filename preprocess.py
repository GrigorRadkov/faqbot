import pickle
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
import numpy as np

pickles = r"D:\Projects\faqbot\cfg\data.pickle"

def preprocess(data):
    #check whether theres new data added to the database, if so retrain the system!

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

    words = [lemmatizer.lemmatize(w) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        #wrds = [lemmatizer.lemmatize(w) for w in doc]
        wrds = [stemmer.stem(w) for w in doc]

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
    
    with open(pickles, "wb") as f:
        pickle.dump((words, labels, training, output), f)