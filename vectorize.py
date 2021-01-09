import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
import numpy as np


def user_input_to_bag_of_words(inp, words):

    bag = [0 for _ in range(len(words))]
    
    input_words = nltk.word_tokenize(inp)
    input_words = [lemmatizer.lemmatize(word.lower()) for word in input_words]
    #input_words = [stemmer.stem(word.lower()) for word in input_words]

    for se in input_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)