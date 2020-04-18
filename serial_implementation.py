import numpy as np


class LDA:

    def __init__(self, num_topics, alpha=None, beta=None, max_iter=20, random_state=205):
        self.num_topics = num_topics
        self.alpha = alpha if alpha else 1 / num_topics
        self.beta = beta if beta else 1 / num_topics
        self.max_iter = max_iter

        self.vocab_size = None
        self.beta_sum = None

        self.topic2cnt = None
        self.word2topic2cnt = None
        self.doc2topic2cnt = None
        self.num_docs = None
        self.doc2word2topic2cnt = None

        np.random.seed(random_state)

    def calculate_mass(self, n_doc, n_word, n_all):
        return (n_doc + self.alpha) * (n_word + self.beta) / (n_all + self.beta_sum)

    def compute_perplexity(self):
        pass

    def fit(self, doc2word2cnt):

        self.num_docs = len(doc2word2cnt)

        # number of words assigned to topic z across all documents
        self.topic2cnt = {i: 0 for i in range(self.num_topics)}

        # number of times each word is assigned to each topic across all documents
        self.word2topic2cnt = {}

        # number of words in each document that are assigned to each topic
        self.doc2topic2cnt = {doc_num: {i: 0 for i in range(self.num_topics)} for doc_num in range(self.num_docs)}

        # number of times each word is assigned to each topic in each document
        self.doc2word2topic2cnt = {i: {} for i in range(self.num_docs)}

        # randomly initialize word assignments
        for doc_id, word2cnt in doc2word2cnt.items():
            for word, word_cnt in word2cnt.items():

                self.doc2word2topic2cnt[doc_id][word] = {i: 0 for i in range(self.num_topics)}

                if word not in self.word2topic2cnt:
                    self.word2topic2cnt[word] = {i: 0 for i in range(self.num_topics)}

                # get random topics

                random_topics = np.random.choice(self.num_topics, word_cnt)

                for random_topic in random_topics:

                    # store topic assignment of word
                    self.doc2word2topic2cnt[doc_id][word][random_topic] += 1

                    # update counts needed for updates
                    self.doc2topic2cnt[doc_id][random_topic] += 1
                    self.topic2cnt[random_topic] += 1
                    self.word2topic2cnt[word][random_topic] += 1

        self.vocab_size = len(self.word2topic2cnt)
        self.beta_sum = self.vocab_size * self.beta

        # perform collapsed Gibbs sampling
        iter_num = 0
        while iter_num < self.max_iter:
            print(iter_num)
            for doc_id, word2cnt in doc2word2cnt.items():
                for word, word_cnt in word2cnt.items():
                    topic2cnt = self.doc2word2topic2cnt[doc_id][word]
                    previous_topics = []
                    for topic, cnt in topic2cnt.items():
                        if cnt > 0:
                            previous_topics.extend([topic] * cnt)

                    np.random.shuffle(previous_topics)

                    for previous_topic in previous_topics:

                        # decrement counts
                        self.doc2word2topic2cnt[doc_id][word][previous_topic] -= 1
                        self.doc2topic2cnt[doc_id][previous_topic] -= 1
                        self.topic2cnt[previous_topic] -= 1
                        self.word2topic2cnt[word][previous_topic] -= 1

                        # assign word to new topic
                        topic_masses = np.array([self.calculate_mass(self.doc2topic2cnt[doc_id][topic_idx],
                                                                     self.word2topic2cnt[word][topic_idx],
                                                                     self.topic2cnt[topic_idx])
                                                 for topic_idx in range(self.num_topics)])
                        topic_masses_norm = topic_masses / np.sum(topic_masses)

                        new_topic = np.random.choice(self.num_topics, 1, p=topic_masses_norm)[0]

                        # increment counts
                        self.doc2word2topic2cnt[doc_id][word][new_topic] += 1
                        self.doc2topic2cnt[doc_id][new_topic] += 1
                        self.topic2cnt[new_topic] += 1
                        self.word2topic2cnt[word][new_topic] += 1

            iter_num += 1

    def get_topic_distributions(self):
        matrix = np.zeros((self.num_topics, self.vocab_size))
        for topic in range(self.num_topics):
            for word in range(self.vocab_size):
                matrix[topic, word] = ((self.word2topic2cnt[word][topic] + self.beta) /
                                       (self.topic2cnt[topic]) + self.beta_sum)

        return matrix

    def get_document_distributions(self):
        matrix = np.zeros((self.num_docs, self.num_topics))
        for doc_id in range(self.num_docs):
            topic2cnt = self.doc2topic2cnt[doc_id]
            topic_assigned_num = sum([topic2cnt[topic] > 0 for topic in range(self.num_topics)])
            for topic in range(self.num_topics):
                matrix[doc_id, topic] = ((self.doc2topic2cnt[doc_id][topic] + self.alpha) /
                                         (topic_assigned_num + self.num_topics * self.alpha))

        return matrix
