from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import re
import os
import boto3

try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'reference','references'}
stop_words |= domain_specific_stop_words

conf = SparkConf().setAppName('logs')
sc = SparkContext.getOrCreate(conf=conf)

#sc = SparkSession.builder.appName("Trial").config("spark.sql.warehouse.dir", "s3://project-cs205-pantent-lda").enableHiveSupport().getOrCreate()

min_doc_length = 10
min_word_occurrences = 5
min_document_occurrences = 1

# need this function to iterate over bucket items
def iterate_bucket_items(bucket):
    """
    Generator that iterates over all objects in a given s3 bucket

    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2
    for return data format
    :param bucket: name of s3 bucket
    :return: dict of metadata for an object
    """
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket)

    for page in page_iterator:
        if page['KeyCount'] > 0:
            for item in page['Contents']:
                yield item

#directory = 'patents_partial1/'
bucket_name = 'cs205-lda-patents'

#files = []
#for i in iterate_bucket_items(bucket=bucket_name):
#    file_name = i['Key']#[len(directory):]
    #if file_name[-4:] == '.nlp':
    #    files.append(file_name)
#    print(file_name)

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

for i, file_dict in enumerate(iterate_bucket_items(bucket=bucket_name)):
    filename = file_dict['Key']
    doc = sc.textFile('s3://'+ bucket_name + '/' + filename).map(get_patent_words)
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

with open('s3://' + bucket_name + '/' + 'word2num_patents.txt', 'w') as f:
    f.write(json.dumps(word2num))

with open('s3://' + bucket_name + '/' + 'doc_id2counts_patents.txt', 'w') as f:
    f.write(json.dumps(doc_id2counts))

#documents_counts.saveAsTextFile("DocumentsCounts.txt")
