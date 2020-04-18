from collections import Counter
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


conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext(conf=conf)

min_doc_length = 10
min_word_occurrences = 5
min_document_occurrences = 3


def preprocess(document):
    # remove symbolic characters and lowercase text
    txt = re.sub("[(),\[\]{}!.?+_:;'$`&]", '', document.lower()).replace('-', ' ')

    # remove extra white space
    txt = re.sub(' +', ' ', txt)

    # remove stopwords and any words with digits
    words = [word for word in txt.split() if word not in stop_words and not any(char.isdigit() for char in word)]
    return words


documents_counts = (sc.textFile('ap.txt')
                    .map(lambda x: x.lstrip())
                    .filter(lambda x: not x.startswith('<') and len(x.split()) > min_doc_length)
                    .map(preprocess)
                    .map(lambda x: Counter(x).most_common()))

documents_counts.cache()

# get words that occur at least 'min_word_occurrences' times throughout all documents and occur in at least
# 'min_document_occurrences' different documents
words_in_vocab = (documents_counts.flatMap(lambda doc: [(key, (value, 1)) for key, value in doc])
                  .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
                  .filter(lambda x: x[1][0] >= min_word_occurrences and x[1][1] >= min_document_occurrences))

word2num = {word: i for i, (word, _) in enumerate(words_in_vocab.collect())}

document_counts_numerical = documents_counts.map(lambda doc_counts: {word2num[word]: cnt for word, cnt in doc_counts
                                                                     if word in word2num})

doc_id2counts = {i: count_dict for i, count_dict in enumerate(document_counts_numerical.collect())}

# num2word = {v: k for k, v in word2num.items()}
# a = doc_id2counts[2240]
# print({num2word[num]: cnt for num, cnt in a.items()})




