import multiprocessing as mp
import sys
import time
import nltk
from nltk.corpus import stopwords
import os
import re

try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'reference','references'}
stop_words |= domain_specific_stop_words

output_file = 'all_patent_docs.txt'


def extract_words(filename, q):
    s = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.lstrip()
            split_line = line.split('\t')
            if len(split_line) > 1:
                # words are on position 1
                word = split_line[1]
                word = re.sub("[(),\[\]{}!.?@+_:;'$`&]", '', word.lower()).replace('-','')
                if not (word in stop_words or len(word) < 4 or any(char.isdigit() for char in word)):
                    s += word + ' '

    s = s.rstrip()
    q.put(s)

def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open(output_file, 'w') as f:
        while True:
            m = q.get()
            if m == 'kill':
                break
            f.write(str(m) + '\n')
            f.flush()


#must use Manager queue here, or will not work
t0 = time.perf_counter()
manager = mp.Manager()
q = manager.Queue()    
pool = mp.Pool()

#put listener to work first
watcher = pool.apply_async(listener, (q,))

# nlp_file_extension = ".nlp"
# nlp_file_extension_len = len(nlp_file_extension)

patent_filenames = []
for directory in os.listdir():
    if directory.startswith('patents'):
        patent_filenames.extend([(directory + '/' + filename, q) for filename in os.listdir(directory) if filename.endswith('.nlp')])

pool.starmap(extract_words, patent_filenames)

print(time.perf_counter() - t0)

#fire off workers
#jobs = []
#for i in range(80):
    #job = pool.apply_async(worker, (i, q))
    #jobs.append(job)

# collect results from the workers through the pool result queue
# for job in jobs: 
#     job.get()

#now we are done, kill the listener
q.put('kill')
pool.close()
pool.join()
