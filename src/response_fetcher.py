import random

def fetch_response(probability, tag, data):
        try:
            #Probability threshold. If the model isn't confident above 70%, that the given intent is the correct one, then it will send a Ambiguity message.
            if  probability > 0.7: 
                #Finds the tag and returns a prewritten response from the knowledge base.
                for intent_tag in data["intents"]:
                    if intent_tag['tag'] == tag:
                        responses = intent_tag['responses']
                return (random.choice(responses))

            else:
                #Ambiguity message
                return "Sorry, I don't quite understand... Can you rephrase that? :)"

        except:
            #In case something blows up.
            return "Error with query. Please try again."