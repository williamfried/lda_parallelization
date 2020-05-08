#!/usr/bin/env python3

import multiprocessing as mp
import sys
import time
import nltk
from nltk.corpus import stopwords
import boto3

try:
    nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('stopwords')
finally:
    stop_words = set(stopwords.words('english'))

domain_specific_stop_words = {'reference','references'}
stop_words |= domain_specific_stop_words

output_file = 'all_patent_docs.txt'
bucket_name = 'cs205-lda-patents'

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


def extract_words(filename, q):
    s = ''
    with open('s3://'+ bucket_name + '/' + filename, 'r') as f:
        for line in f:
            line = line.lstrip()
            split_line = line.split('\t')
            if len(split_line) > 1:
                # words are on position 1
                word = split_line[1]
                word = re.sub("[(),\[\]{}!.?@+_:;'$`&]", '', word.lower()).replace('-','')
                if not (word in stop_words or len(word) < 4 or any(char.isdigit() for char in word)):
                    s += ' ' + word

    q.put(s)

def listener(q):
    '''listens for messages on the q, writes to file. '''

    with open('s3://'+ bucket_name + '/' + output_file, 'w') as f:
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

filenames = [(file_dict['Key'], q) for file_dict in iterate_bucket_items(bucket_name)]

print(len(filenames))
print(time.perf_counter() - t0)
sys.exit(2)


pool.starmap(extract_words, filenames)

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
