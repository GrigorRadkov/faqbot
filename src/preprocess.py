import pickle
import json
import nltk
import os
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

#Creates lemmatizer
lemmatizer = WordNetLemmatizer()

#Necessary paths
curr_dir = os.getcwd()
pickles = "/cfg/data.pickle"

def preprocess(data):

    #Temporary lists needed for preprocessing the knowledge base data.
    #The words list is used in intent_predictor.py in order to construct a vector of the user input.
    words = []
    #Patterns_list will be a list of lists. Every unique sentence from the knowledge base is contained in a separate list.
    patterns_list = []
    #Labels_list is a single list that contains all the intent tags from the knowledge base. They correspond to lists in the doc_x list. 
    #I.e. patterns_list[0] will be a sentence, whose intent is in labels_list[0].
    labels_list = []
    #Labels is a single list that contains all unique labels.
    labels = []
    #The two lists used in the nn_model_handler.py. They are used to fit our neural network model with our now processed data.
    train_data = []
    target_data = []

    #Fills the patterns_list and words lists with all tokenized questions/patterns and the labels_list file with the corresponding labels.
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            temp_words = nltk.word_tokenize(pattern)
            temp_words = [token.lower() for token in temp_words if token.isalpha()]
            words.extend(temp_words)
            patterns_list.append(temp_words)
            labels_list.append(intent["tag"])
        #Appends the intent tags in the labels list. This is needed for the intent_predictor.py function, in order to find the most likely label/intent of the user.
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    labels = sorted(labels)

    #Every word is lemmatized/stemmed. Helps keep the knowledge base smaller.
    for word, tag in nltk.pos_tag(words):
        words.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

    #Sorting isn't necessary, but it helps when playing with the code.
    words = sorted(list(set(words)))

    for i, sentence in enumerate(patterns_list):
        #List used to store the bag of one hot encoded word vectors.
        bag_of_words = []

        #Creates a temprorary list to store all lemmas of every sentence list in the patterns_list.
        temp_lemmas = []
        for word, tag in nltk.pos_tag(sentence):
            temp_lemmas.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

        #Creates the vector for a sentence by appending a 1, where a word appears in both words and temp_lemmas lists, and a 0 otherwise.
        #This means a single sentence's vector is of the same length as the words list.
        for word in words:
            if word in temp_lemmas:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        #Appends the bow for the sentence to the list of sentences.
        train_data.append(bag_of_words)
        
        #Creates the one hot encoded vector for the labels. And appends it to the list of label targets.
        target_data_row = np.zeros((len(labels), ), dtype=int)
        target_data_row[labels.index(labels_list[i])] = 1
        target_data.append(target_data_row)
    
    #Convert to numpy arrays, as tensorflow requires it.
    train_data = np.array(train_data)
    target_data = np.array(target_data)

    #Save data to a pickles dump
    with open(curr_dir + pickles, "wb") as f:
        pickle.dump((words, labels, train_data, target_data), f)