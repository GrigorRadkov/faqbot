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

intents = r"D:\Projects\faqbot\intents\intents.json"
pickles = r"D:\Projects\faqbot\cfg\data.pickle"

client = commands.Bot(command_prefix = '!')

@client.event
async def on_ready():
    print('In operation')
    print(client.user.id)

@client.command()
async def ping(ctx):
    await ctx.send(f'Pong! {round(client.latency * 1000)}ms')

@client.command(hidden = True)
async def stopQ(ctx):
    await client.logout()

@client.command()
async def check(ctx, *args):
    await ctx.send('Remember to write sentences in brackets. `{}` arguments: `{}`'.format(len(args), ', '.join(args)))

@client.command(hidden = True)
async def add(ctx, tag, question, response):
    print('invoked')
    flag = add_to_json(tag, question, response)

    if(flag == 1):
        await ctx.send(f'Question added to database: tag - {tag}, question - {question}, response - {response}')
    if(flag == 2):
        await ctx.send('Failed to add question to database')

@client.event
async def on_message(message):

    #Bot talks only if mentioned
    if  message.content.find("<@!707633269755084812>") != -1 and message.author != client.user:

        query = message.content

        print(query)

        probability, tag = predict_intent(query, model, words, labels, data)
        response = fetch_response(probability, tag, data)
        
        async with message.channel.typing():
            await asyncio.sleep(random.randint(1,3))

        await message.channel.send("{}".format(message.author.mention) + ' ' + response, tts = True)

    await client.process_commands(message)
try:
    with open(intents) as file:
        data = json.load(file)
except:
    print("Error loading intents file")

try:
    with open(pickles, "rb") as f:
        words, labels, training, output = pickle.load(f)      
        print(training)  
except:
    preprocess(data)
    print('1')
    with open(pickles, "rb") as f:
        words, labels, training, output = pickle.load(f)
try:
    model = load_model(training, output)
    print('2')
except:
    model = create_model(training, output)
    print('3')

client.run(token)