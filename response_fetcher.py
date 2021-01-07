import random

def fetch_response(probability, tag, data):
        try:
            if  probability > 0.7: 
            
                for tg in data["intents"]:
            
                    if tg['tag'] == tag:
                        responses = tg['responses']
                return (random.choice(responses))
            else:
                return "Sorry, I don't quite understand... Can you rephrase that? :)"
        except:
            
            return "Error with query. Please try again."