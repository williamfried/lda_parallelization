import json
import numpy as np
from mpi_imp import LDA
import time

top_word_num = 30

with open('doc_id2counts.txt', 'r') as f:
    doc_id2counts_ = json.loads(f.read())

with open('word2num.txt', 'r') as f:
    word2num = json.loads(f.read())

doc_id2counts = {}
for doc_id, word2cnt in doc_id2counts_.items():
    doc_id2counts[int(doc_id)] = {int(word): cnt for word, cnt in word2cnt.items()}

num2word = {v: k for k, v in word2num.items()}

num_topics = 50
lda = LDA(num_topics)
t0 = time.perf_counter()
lda.fit(doc_id2counts)
print(time.perf_counter() - t0)
topic_distributions = lda.get_topic_distributions()
document_distributions = lda.get_document_distributions()

topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)

top_words = np.vectorize(num2word.get)(topic_distributions_sorted)[:, :top_word_num]

for i in range(num_topics):
    print(i)
    print(top_words[i, :])
    print()

document_distributions_sorted = np.argsort(-document_distributions, axis=1)
for i in range(10):
    print(document_distributions_sorted[i, :5])
    print()
