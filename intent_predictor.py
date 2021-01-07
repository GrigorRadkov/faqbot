import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import *
from vectorize import *

def predict_intent(query, model, words, labels, data):

        tmp = [user_input_to_bag_of_words(query, words)]
        padded = pad_sequences(tmp)
        probabilities = model.predict(padded)
        best_probability_index = np.argmax(probabilities)
        best_probability = probabilities[0][best_probability_index]

        tag = labels[best_probability_index]

        return best_probability, tag


        