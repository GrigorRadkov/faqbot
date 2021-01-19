import timeit

setup = """
import nltk
import sys
sys.path.insert(1, r"D:/Projects/faqbot/src")
import wordnet_pos_converter
from nltk.stem.wordnet import WordNetLemmatizer
from wordnet_pos_converter import get_wordnet_pos

lemmatizer = WordNetLemmatizer()

words = []
text = []
tmp = []

text = "A random sentence."

temp_words = nltk.word_tokenize(text)
temp_words = [token.lower() for token in temp_words if token.isalpha()]
words.extend(temp_words)
"""

print(timeit.timeit("""
for word, tag in nltk.pos_tag(words):
    tmp.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag[0])))
""", setup = setup, number = 1))
