import pickle
import json
import nltk
import os
import numpy as np

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

#stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

#Necessary paths
curr_dir = os.getcwd()
pickles = "/cfg/data.pickle"

def preprocess(data):

    #Temporary lists needed for preprocessing the knowledge base data.
    #The words list is used in intent_predictor.py in order to construct a vector of the user input
    words = []
    #Doc_x will be a list of lists. Every unique sentence from the knowledge base is contained in a separate list.
    docs_x = []
    #Doc_y is a single list that contains all the intent tags from the knowledge base. They correspond to lists in the doc_x list. 
    #I.e. docs_x[0] will be a sentence, whose intent is in docs_y[0].
    docs_y = []
    #Labels is a single list of all labels.
    labels = []

    #Fills the docs_x and words lists with all tokenized questions/patterns and the docs_y file with the corresponding labels.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp_words = nltk.word_tokenize(pattern)
            temp_words = [token.lower() for token in temp_words if token.isalpha()]
            words.extend(temp_words)
            docs_x.append(temp_words)
            docs_y.append(intent["tag"])
        #Appends the intent tags in the labels list. This is needed for the intent_predictor.py function, in order to find the most likely label/intent of the user.
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #Every word is lemmatized/stemmed. Helps keep the knowledge base smaller.
    #words = [stemmer.stem(w.lower()) for w in words if w.isalpha()] If you want to use stemming. Just comment the 2 lines below and use this line.
    for word, tag in nltk.pos_tag(words):
        words.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

    #Sorting isn't necessary, but it helps when playing with the code.
    words = sorted(list(set(words)))
    labels = sorted(labels)

    #The two lists used in the nn_model_handler.py. They are used to fit our neural network model with our now processed data.
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        #Creates a temprorary list to store all lemmas of every sentence list in doc_x. Appends
        temp_lemmas = []
        for word, tag in nltk.pos_tag(doc):
            temp_lemmas.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

        #temp_stems = [stemmer.stem(w.lower()) for w in doc] For stemming use this line and comment the 3 lines above this.

        for w in words:
            if w in temp_lemmas:
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