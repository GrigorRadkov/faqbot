import pickle
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np



def preprocess(data):
    #check whether theres new data added to the database, if so retrain the system!
    try:
        with open("D:\Ot uni\Diplomna\FAQBot\cfg\data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:

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

        words = [stemmer.stem(w) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

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
        
        with open("D:\Ot uni\Diplomna\FAQBot\cfg\data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)