import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vectorize import user_input_to_bag_of_words

def predict_intent(query, model, words, labels, data):

        #Creates the user input vector and pad it.
        tmp = [user_input_to_bag_of_words(query, words)]
        padded = pad_sequences(tmp)
        #Uses the neural network model to predict the user's intent based on his now vectorized input.
        probabilities = model.predict(padded)
        #Finds the best probability and it's index.
        best_probability_index = np.argmax(probabilities)
        best_probability = probabilities[0][best_probability_index]
        #Finds the tag that represents the index in the labels list.
        tag = labels[best_probability_index]

        return best_probability, tag


        