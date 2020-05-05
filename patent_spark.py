from collections import Counter
import json
import nltk
from nltk.corpus import stopwords
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import re
import os
import boto3


#!/usr/bin/python3
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import pyspark.sql.functions as F

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
min_document_occurrences = 3

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

bucket_name = 'cs205-lda-patents'
#bucket_name = "test-docs-cs-205-lda"


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


"""

After this loop, we'll have an RDD where the key is the document number, and the value is a list of all
the words in the document. It looks something like this:

('doc1', ['application', 'november', 'serial', 'claim'])
('doc2', ['wku', 'pdid', 'us', 'pttl', 'ocr', 'scanned', 'documentdsrc', 'oclpar', 'april', 'nicolas', 'rose',
'assignor', 'jack', 'son', 'perkins', 'company', 'newark'])
"""
documents_counts = None
document_counter = 0
for file_dict in iterate_bucket_items(bucket=bucket_name):
    filename = file_dict['Key']
    if filename[-4:] != ".nlp":
        # Skip over iterating anything that's not a .nlp file
        continue
    doc = sc.textFile('s3://'+ bucket_name + '/' + filename).map(get_patent_words)
    doc = doc.map(lambda word: ('doc' + str(document_counter), word))
    doc = doc.reduceByKey(lambda a,b: a+b)
    if not documents_counts:
        documents_counts = doc
    else:
        documents_counts = documents_counts.union(doc)
    document_counter += 1
    #if document_counter > 5:
    #    break

"""
After the reduceByKey, the output is like:

(('doc5', 'vigorous'), 2)
(('doc5', 'seven'), 1)
(('doc5', 'sizemedium'), 1)
(('doc5', 'distinguishing'), 1)
(('doc5', 'adaptable'), 1)
(('doc5', 'colornew'), 1)
(('doc5', 'foliageupper'), 2)
(('doc5', 'sidebronzy'), 2)
(('doc5', 'purposes'), 1)
(('doc5', 'substantially'), 1)

After the entire process, the result is like:
('doc3', [('pttl', 1), ('ocr', 1), ('oclpar', 1), ('nicolas', 3), ('filed', 1), ('nov', 1),
('pat', 2), ('patented', 1), ('states', 1), ('patent', 1), ('office', 1), ('jean', 2), ('newark', 3),
('assignor', 1), ('jack', 1), ('perkins', 1), ('company', 1), ('application', 1), ('november', 1),
"""
documents_counts = documents_counts.flatMap(
        lambda doc: [((doc[0], word), 1) for word in doc[1]]
    ).reduceByKey(lambda prev, next: prev + next
    ).map(lambda line: (line[0][0], [(line[0][1], line[1])])
    ).reduceByKey(lambda prev, next: prev + next
)


"""
get words that occur at least 'min_word_occurrences' times throughout all
documents and occur in at least 'min_document_occurrences' different documents
"""
words_in_vocab = (documents_counts.flatMap(
        lambda doc : [(key, (value, 1)) for key, value in doc[1]]
    ).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])
    ).filter(lambda x: x[1][0] >= min_word_occurrences and x[1][1] >= min_document_occurrences)
)


"""
words_in_vocab should now contain:

('peach', (72, 4))
('color', (68, 6))
('new', (60, 6))
('size', (56, 6))
('tree', (52, 4))
('medium', (44, 4))
('rose', (44, 4))
('plant', (36, 6))
('long', (32, 6))
('length', (28, 6))
('freestone', (24, 4))
"""

# map each word to an arbitrary integer between 0 and vocab_size - 1
word2num = {word: i for i, (word, _) in enumerate(words_in_vocab.collect())}
word2numRDD = words_in_vocab.map(lambda tup:tup[0]).zipWithIndex()
# save word2num
word2numRDD.saveAsTextFile('s3://' + bucket_name + '/' + 'word2num_patents.txt')

# filter out words that aren't in the vocabulary and convert words that are in the vocabulary to the corresponding
# integers
# map each document (represented by an integer between 0 and document_number - 1) to a dictionary that maps each word
# to the number of times it appears in the given document
document_counts_numerical = documents_counts.map(lambda doc_counts: {word2num[word]: cnt for word, cnt in doc_counts[1] if word in word2num})

document_counts_numerical = document_counts_numerical.zipWithIndex().map(lambda tup : (tup[1],tup[0]))

document_counts_numerical.saveAsTextFile('s3://' + bucket_name + '/' + 'doc_id2counts_patents.txt')