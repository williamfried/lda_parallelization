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

conf = SparkConf().setMaster('local[*]').setAppName('preprocess-text')
sc = SparkContext(conf=conf)

min_doc_length = 10
min_word_occurrences = 5
min_document_occurrences = 3
num_output_files = 2


def preprocess(document):
    # remove symbolic characters and lowercase text
    txt = re.sub("[(),\\[\\]{}!.?+_:;'\"$`&]", '',
                 document.lower()).replace('-', ' ').replace('/', ' ')
    # remove extra white space
    txt = re.sub(' +', ' ', txt)
    # remove stopwords and any words with digits
    words = [word for word in txt.split()
             if word not in stop_words
             and not any(char.isdigit() for char in word)]
    return words


# get a count of each preprocessed word for each document
documents_counts = sc.textFile('ap.txt').map(lambda x: x.lstrip()).filter(
    lambda x: not x.startswith('<') and len(x.split()) > min_doc_length
).map(preprocess).map(lambda x: Counter(x).most_common())

# cache RDD to avoid having to perform pipeline above multiple times
documents_counts.cache()

# get words that occur at least 'min_word_occurrences' times throughout all
# documents and occur in at least 'min_document_occurrences' different documents
words_in_vocab = documents_counts.flatMap(
    lambda doc: [(key, (value, 1)) for key, value in doc]
).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).filter(
    lambda x:
    x[1][0] >= min_word_occurrences and x[1][1] >= min_document_occurrences
)

# map each word to an arbitrary integer between 0 and vocab_size - 1
word2num = words_in_vocab.map(lambda x: x[0]).zipWithIndex().collectAsMap()

# filter out words that aren't in the vocabulary and convert words that are in
# the vocabulary to the corresponding integers
document_counts_numerical = documents_counts.map(
    lambda doc_counts: [(word2num[word], cnt)
                        for word, cnt in doc_counts if word in word2num]
)

# map each document (represented by an integer between 0
# and document_number - 1) to a string that lists each word with the number of
# times it appears in the given document
doc_id2counts = document_counts_numerical.zipWithIndex().map(
    lambda x: str(x[1]) + "," + ",".join(str(entry)
                                         for token in x[0] for entry in token))

# two objects we need from here:
# 1. word2num to reverse the dictionary and recover the words corresponding to
#    each integer after LDA is performed
# 2. doc_id2counts

with open('word2num.txt', 'w') as f:
    f.write(json.dumps(word2num))

doc_id2counts.coalesce(num_output_files).saveAsTextFile("output")
