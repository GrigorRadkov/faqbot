import random
import numpy as np
from vectorize import *

def predict_intent(query, model, words, labels, data):
        
        probabilities = model.predict([user_input_to_bag_of_words(query, words)])[0]
        best_probability_index = np.argmax(probabilities)
        tag = labels[best_probability_index]
#maybe put the response generation part in another function
        if probabilities[best_probability_index] > 0.7: 

            for tg in data["intents"]:
        
                if tg['tag'] == tag:
                    responses = tg['responses']
            response = (random.choice(responses))
            return response
        else:
            response = "No comprendo, señor"
            return response