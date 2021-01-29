import timeit

setup = """import os
import nltk
import sys
sys.path.insert(1, r"D:/Projects/faqbot/src")
from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

lemmatizer = WordNetLemmatizer()
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
for word, tag in nltk.pos_tag(words):
    tmp.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))
""", setup = setup, number=10000))