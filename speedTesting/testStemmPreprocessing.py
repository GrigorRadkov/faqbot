import timeit

setup = """import os
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
curr_dir = os.getcwd()
file = "/intents/wikitextsample.txt"

words = []
text = []
tmp = []
# open file and read the content in a list
with open(curr_dir + file, 'r') as f:
    text = [current_line.rstrip() for current_line in f.readlines()]
i = 0
while i < 10:
    temp_words = nltk.word_tokenize(text[i])
    temp_words = [token.lower() for token in temp_words if token.isalpha()]
    words.extend(temp_words)
    i += 1
words = sorted(list(set(words)))
"""

print(timeit.timeit("""
tmp = [stemmer.stem(w) for w in words]
""", setup = setup, number = 10000))