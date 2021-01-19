import json
import os

#Necessary paths
curr_dir = os.getcwd()
intents = "/intents/intents.json"

def add_to_json(tag, question, response):
    
    try:
        with open(curr_dir + intents, 'r+') as f:
            data = json.load(f)

            for intent in data['intents']:
                if intent['tag'].lower() != '':
                    if intent['tag'].lower() == tag.lower():
                        if question != '': 
                            if question.capitalize() not in intent['patterns']:
                                intent['patterns'].append(question.capitalize())
                        if response != '':
                            if response.capitalize() not in intent['responses']:
                                intent['responses'].append(response.capitalize())
                    if intent['tag'].lower() != tag.lower():
                        intent['tag'].append(tag.lower())
                        if question != '': 
                            if question.capitalize() not in intent['patterns']:
                                intent['patterns'].append(question.capitalize())
                        if response != '':
                            if response.capitalize() not in intent['responses']:
                                intent['responses'].append(response.capitalize())
            f.seek(0)
            json.dump(data, f, indent = 2)
            f.truncate()
            f.close()
        return 1
    except:
        return 2