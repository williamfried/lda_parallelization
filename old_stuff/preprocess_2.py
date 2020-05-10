import multiprocessing as mp
import sys
import time
import nltk
from nltk.corpus import stopwords
import os
import re

# use standard nltk stopwords
try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'reference', 'references'}
stop_words |= domain_specific_stop_words

output_file = 'all_patent_docs.txt'

def extract_words(filename, q):
    '''Extract valid words from each patent document. The actual words of the patent are positioned in the second column of the text file.
    Valid words are those that are not stopwords, don't contain any numeric digits are at least four characters long. The words are
    added to a string, which is then pushed onto the queue.'''
    s = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.lstrip()
            split_line = line.split('\t')
            if len(split_line) > 1:
                word = split_line[1]
                word = re.sub("[(),\[\]{}!.?@+_:;'$`&]", '', word.lower()).replace('-','')
                if not (word in stop_words or len(word) < 4 or any(char.isdigit() for char in word)):
                    s += word + ' '

    s = s.rstrip()
    q.put(s)

def listener(q):
    '''Listens for messages on the queue and writes a new line to the output file for each patent document.'''

    with open(output_file, 'w') as f:
        while True:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()


# set up queue
manager = mp.Manager()
q = manager.Queue()    

pool = mp.Pool()

# activate listener
watcher = pool.apply_async(listener, (q,))

# get complete list of filenames where the patent documents reside
patent_filenames = []
for directory in os.listdir():
    if directory.startswith('patents'):
        patent_filenames.extend([(directory + '/' + filename, q) for filename in os.listdir(directory) if filename.endswith('.nlp')])

# assign filenames in parallel to each of the worker processes
pool.starmap(extract_words, patent_filenames)


# terminate listener 
q.put('kill')
pool.close()
pool.join()
