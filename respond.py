import random
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
        tag = labels[best_probability_index]

#maybe put the response generation part in another function

        if probabilities[0][best_probability_index] > 0.7: 

            for tg in data["intents"]:
        
                if tg['tag'] == tag:
                    responses = tg['responses']
            response = (random.choice(responses))
            return response
        else:
            response = "Sorry, I don't quite understand... Can you rephrase that? :)"
            return response