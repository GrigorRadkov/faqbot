import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

#Create lemmatizer
lemmatizer = WordNetLemmatizer()

def user_input_to_bag_of_words(inp, words):

    #creates a vector full of zeroes, with the length of all the words in the knowledge base.
    bag_of_words = np.zeros((len(words), ), dtype=int)
    
    temp_lemmas = []

    #Preprocessing the user's input
    input_words = nltk.word_tokenize(inp)
    input_words = [token.lower() for token in input_words if token.isalpha()]
    for word, tag in nltk.pos_tag(input_words):
        temp_lemmas.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))

    #One hot encoding
    for lemma in temp_lemmas:
        for i, word in enumerate(words):
            if word == lemma:
                bag_of_words[i] = 1

    return np.array(bag_of_words)