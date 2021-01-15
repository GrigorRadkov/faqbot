import os
import discord
from discord.ext import commands
import asyncio

import numpy as np
import tensorflow as tf
import random
import json
import pickle

from preprocess import *
from nn_model_handler import *
from intent_predictor import *
from add_to_json import *
from response_fetcher import *
from discord_token import token

#Paths used for creation and access of files
curr_dir = os.getcwd()
intents = "/intents/intents.json"
pickles = "/cfg/data.pickle"
token_file = "/discord_token.py"

#The prefix that users need to use when typing commands in order for the bot to understand them.
client = commands.Bot(command_prefix = '!')

@client.event
async def on_ready():
    print('In operation')
    print(client.user.id)

#Just a latency checker
@client.command()
async def ping(ctx):
    await ctx.send(f'Pong! {round(client.latency * 1000)}ms')

#A command to logout the bot. I.e. stop it.
@client.command(hidden = True)
async def stopQ(ctx):
    await client.logout()

#This command can be used to check how a user input is handled by discord. Each word after the command is a separate argument. 
#If you want a sentence to be a single argument, then write it in "" brackets.
@client.command()
async def check(ctx, *args):
    await ctx.send('Remember to write sentences in brackets. `{}` arguments: `{}`'.format(len(args), ', '.join(args)))

#A command that can be used to add an entry to your knowledge base. I.e. the intents.json.
#Write things as you would for the check command.
@client.command(hidden = True)
async def add(ctx, tag, question, response):

    flag = add_to_json(tag, question, response)

    if(flag == 1):
        await ctx.send(f'Question added to database: tag - {tag}, question - {question}, response - {response}')
    if(flag == 2):
        await ctx.send('Failed to add question to database')

#Standard discord on_message function. Refer to the discord API for information.
@client.event
async def on_message(message):

    #Bot talks only if mentioned
    if  message.content.find("<@!707633269755084812>") != -1 and message.author != client.user:

        query = message.content

        probability, tag = predict_intent(query, model, words, labels, data)
        response = fetch_response(probability, tag, data)
        
        #Pseudo-random delay. Makes the chatbot look like it's typing, which looks more natural, right? Remove this function if you do not want delay.
        async with message.channel.typing():
            await asyncio.sleep(random.randint(1,3))

        #If Text-to-Speech is annoying you, use False instead.
        await message.channel.send("{}".format(message.author.mention) + ' ' + response, tts = True)

    await client.process_commands(message)

#Checks whether the pickles file exists and creates one if not.
if not os.path.exists(curr_dir + pickles):
    open(curr_dir + pickles, 'w')

#Checks whether the token file exists and creates one if not.
if not os.path.exists(curr_dir + token_file):
    open(curr_dir + token_file, 'w')

#Tries to load the knowledge base data.
try:
    with open(curr_dir + intents) as file:
        data = json.load(file)
except:
    print("Intents.json file not found!")

#Tries to load the pickles file, if the chatbot hasn't been ran at least once, 
#it will begin preprocessing the knowledge base data and save the necessary data as a .pickles file, for later use.
try:
    with open(curr_dir + pickles, "rb") as f:
        words, labels, training, output = pickle.load(f)      
except:
    preprocess(data)
    with open(curr_dir + pickles, "rb") as f:
        words, labels, training, output = pickle.load(f)

#Tries to load the neural network model, if the chatbot hasn't been ran at least once,
#it will initiate the neural-network handler to create a model, based on the data saved in the .pickles file.
try:
    model = load_model(training, output)
except:
    model = create_model(training, output)

#Tries to authenticate the bot, using the token found in the discord_token.py file.
try:
    client.run(token)
except:
    print("Discord token could not be retrieved. Please open the discord_token.py file in the chatbot's folder and write the following \ntoken = 'tokencode', where 'tokencode' refers to the discord token for your bot, found on the discord developer website: https://discord.com/developers/applications")