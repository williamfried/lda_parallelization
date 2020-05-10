#!/usr/bin/python3
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local[1]').setAppName('logs')
sc = SparkContext.getOrCreate(conf=conf)

bucket_name = 'cs205-lda-patents'
min_doc_length = 10
min_word_occurrences = 5
min_document_occurrences = 5

"""

Throughout the following RDD operations the data will be reorganized as:

### 1
('doc1', ['application', 'november', 'serial', 'claim'])
('doc2', ['wku', 'pdid', 'us', 'pttl', 'ocr', 'scanned', 'documentdsrc', 'oclpar', 'april', 'nicolas', 'rose',
'assignor', 'jack', 'son', 'perkins', 'company', 'newark'])

### 2
(('doc5', 'vigorous'), 2)
(('doc5', 'seven'), 1)
(('doc5', 'adaptable'), 1)
(('doc5', 'substantially'), 1)

### 3
('doc3', [('oclpar', 1), ('nicolas', 3), ('filed', 1), ('nov', 1),('pat', 2), ('patented', 1),
('states', 1), ('patent', 1), ('office', 1), ('jean', 2), ('newark', 3),('assignor', 1), ('jack', 1),
('perkins', 1), ('company', 1), ('application', 1), ('november', 1)])

"""

documents_counts = (sc.textFile('all_patent_docs.txt')
                    .map(lambda doc_line: doc_line.split())
                    .zipWithIndex()
                    .flatMap(lambda tup: [((tup[1],str(word)),1) for word in tup[0]])
                    .reduceByKey(lambda prev, nextt: prev + nextt)
                    .map(lambda line: (line[0][0], [(line[0][1], line[1])]))
                    .reduceByKey(lambda prev, nextt: prev + nextt))
                 
documents_counts = documents_counts.repartition(40)

"""
get words that occur at least 'min_word_occurrences' times throughout all
documents and occur in at least 'min_document_occurrences' different documents
"""
words_in_vocab = (documents_counts.flatMap(
        lambda doc: [(key, (value, 1)) for key, value in doc[1]]
    ).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])
    ).filter(lambda x: x[1][0] >= min_word_occurrences and x[1][1] >= min_document_occurrences)
)

'''
words_in_vocab should now contain:
#
# ('peach', (72, 4))
# ('color', (68, 6))
# ('new', (60, 6))
# ('size', (56, 6))
# ('tree', (52, 4))
# ('medium', (44, 4))
# ('rose', (44, 4))
# ('plant', (36, 6))
# ('long', (32, 6))
# ('length', (28, 6))
# ('freestone', (24, 4))
'''

# map each word to an arbitrary integer between 0 and vocab_size - 1
words_in_vocab.saveAsTextFile('s3://' + bucket_name + '/' + 'vocab_serial2.txt')
word2num = {word: i for i, (word, _) in enumerate(words_in_vocab.collect())}
word2numRDD = words_in_vocab.map(lambda tup: tup[0]).zipWithIndex()
# save word2num
word2numRDD.saveAsTextFile('s3://' + bucket_name + '/' + 'word2num_patents_seria2.txt')

# filter out words that aren't in the vocabulary and convert words that are in the vocab to the corresponding integers
# map each document (represented by an integer between 0 and document_number - 1) to a dictionary that maps each word
# to the number of times it appears in the given document
document_counts_numerical = documents_counts.map(lambda doc_counts: {word2num[word]: cnt for word, cnt in doc_counts[1]
                                                                     if word in word2num})

document_counts_numerical = document_counts_numerical.zipWithIndex().map(lambda tup: (tup[1], tup[0]))
document_counts_numerical.saveAsTextFile('s3://' + bucket_name + '/' + 'doc_id2counts_patents_serial2.txt')
