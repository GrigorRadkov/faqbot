import timeit

setup2 = """
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

words = []
text = []
tmp = []

text = "A random sentence."

temp_words = nltk.word_tokenize(text)
temp_words = [token.lower() for token in temp_words if token.isalpha()]
words.extend(temp_words)
"""

print(timeit.timeit("""
tmp = [stemmer.stem(w) for w in words]
""", setup = setup2, number = 1))