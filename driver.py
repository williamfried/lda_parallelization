import json
import numpy as np
from serial_implementation import LDA
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

num_topics = 20
lda = LDA(num_topics, max_iter=20)
t0 = time.perf_counter()
lda.fit(doc_id2counts)
print(time.perf_counter() - t0)
topic_distributions = lda.get_topic_distributions()
document_distributions = lda.get_document_distributions()

topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)

top_words = np.vectorize(num2word.get)(topic_distributions_sorted)[:, :top_word_num]

for i in range(num_topics):
    print(top_words[i, :])
    print()


# a = np.array([[1,4,6,2,8,3] * 10, [7,4,6,9,2,1] * 10])
# highest = np.argsort(-a, axis=1)
# for i in range(2):
#     print(highest[i, :])
# print(highest)