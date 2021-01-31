import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

lemmatizer = WordNetLemmatizer()

def user_input_to_bag_of_words(inp, words):

    #creates a vector full of zeroes, with the length of all the words in the knowledge base.
    bag = [0 for _ in range(len(words))]
    
    tmp = []

    #Preprocessing the user's input
    input_words = nltk.word_tokenize(inp)
    input_words = [token.lower() for token in input_words if token.isalpha()]
    for word, tag in nltk.pos_tag(input_words):
        tmp.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

    #One hot encoding
    for se in tmp:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)