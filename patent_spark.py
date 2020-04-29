from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
import re
import os

try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'reference','references'}
stop_words |= domain_specific_stop_words

conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext.getOrCreate(conf=conf)

min_doc_length = 10
min_word_occurrences = 5
min_document_occurrences = 1

directory = 'patent_data'

def get_patent_words(x):
    words = []
    line = x.lstrip()
    split_line = line.split('\t')
    if len(split_line)>1:
        # words are on position 1
        word = split_line[1]
        word = re.sub("[(),\[\]{}!.?+_:;'$`&]", '', word.lower()).replace('-','')
        valid = True
        if word in stop_words:
            valid = False
        if len(word) < 2:
            valid = False
        if any(char.isdigit() for char in word):
            valid = False
        if valid:
                words.append(word)
    return words

for i, filename in enumerate(os.listdir(directory)):
    doc = sc.textFile(directory + '/' + filename).map(get_patent_words)
    doc = doc.map(lambda word: ('doc'+str(i), word))
    doc = doc.reduceByKey(lambda a,b: a+b)
    if i == 0:
        documents_counts = doc
    else:
        documents_counts = documents_counts.union(doc)
    
documents_counts = documents_counts.map(lambda tup: Counter(tup[1]).most_common())

# get words that occur at least 'min_word_occurrences' times throughout all documents and occur in at least
# 'min_document_occurrences' different documents
words_in_vocab = (documents_counts.flatMap(lambda doc: [(key, (value, 1)) for key, value in doc])
                  .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
                  .filter(lambda x: x[1][0] >= min_word_occurrences and x[1][1] >= min_document_occurrences))

# map each word to an arbitrary integer between 0 and vocab_size - 1
word2num = {word: i for i, (word, _) in enumerate(words_in_vocab.collect())}

# filter out words that aren't in the vocabulary and convert words that are in the vocabulary to the corresponding
# integers
document_counts_numerical = documents_counts.map(lambda doc_counts: {word2num[word]: cnt for word, cnt in doc_counts if word in word2num})

# map each document (represented by an integer between 0 and document_number - 1) to a dictionary that maps each word
# to the number of times it appears in the given document
doc_id2counts = {i: count_dict for i, count_dict in enumerate(document_counts_numerical.collect())}

# two objects we need from here:
# 1. word2num to reverse the dictionary and recover the words corresponding to each integer after LDA is performed
# 2. doc_id2counts

with open('word2num_patents.txt', 'w') as f:
    f.write(json.dumps(word2num))

with open('doc_id2counts_patents.txt', 'w') as f:
    f.write(json.dumps(doc_id2counts))

documents_counts.saveAsTextFile("DocumentsCounts.txt")
