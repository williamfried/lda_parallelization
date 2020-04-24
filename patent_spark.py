from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
import re

try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'said'}
stop_words |= domain_specific_stop_words

conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext(conf=conf)

min_doc_length = 10
min_word_occurrences = 5

def preprocess(document):
    # remove symbolic characters and lowercase text
    txt = re.sub("[(),\[\]{}!.?+_:;'$`&]", '', document.lower()).replace('-', ' ')

    # remove extra white space
    txt = re.sub(' +', ' ', txt)

    # remove stopwords and any words with digits
    words = [word for word in txt.split() if word not in stop_words and not any(char.isdigit() for char in word)]
    return words


conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext(conf=conf)

test = sc.textFile('02444358.nlp')

lines = test.map(lambda x: str(x.split('\t')[1]) if len(x.split('\t')) > 1 else '') 

lines.saveAsTextFile("output.txt")
