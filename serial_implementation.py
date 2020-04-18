import numpy as np


class LDA:

    def __init__(self, num_topics, alpha, beta, max_iter=10, random_state=205):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.beta_sum = np.sum(beta)
        self.max_iter = max_iter
        self.topic2cnt = None
        self.word2topic2cnt = None
        self.doc2topic2cnt = None
        self.num_docs = None
        self.doc2word_idx2topic = None

        np.random.seed(random_state)

    def calculate_mass(self, n_doc, n_word, n_all, topic_idx, word):
        return (n_doc + self.alpha[topic_idx]) * (n_word + self.beta[word]) / (n_all + self.beta_sum)

    def compute_perplexity(self):
        pass

    def fit(self, documents):

        self.num_docs = len(documents)

        # number of words assigned to topic z across all documents
        self.topic2cnt = {i: 0 for i in range(self.num_topics)}

        # number of times word w is assigned to topic z across all documents
        self.word2topic2cnt = {}

        # number of words in document d that are assigned to topic z
        self.doc2topic2cnt = {doc_num: {i: 0 for i in range(self.num_topics)} for doc_num in range(self.num_docs)}

        self.doc2word_idx2topic = {i: {} for i in range(self.num_docs)}

        # randomly initialize word assignments
        for doc_idx, document in enumerate(documents):
            for word_idx, word in enumerate(document):

                # get random topic
                random_topic = np.random.randint(0, self.num_topics, 1)[0]

                # store topic assignment of word
                self.doc2word_idx2topic[doc_idx][word_idx] = random_topic

                # update counts needed for updates
                self.doc2topic2cnt[doc_idx][random_topic] += 1
                self.topic2cnt[random_topic] += 1
                if word not in self.word2topic2cnt:
                    self.word2topic2cnt[word] = {i: 0 for i in range(self.num_topics)}
                self.word2topic2cnt[word][random_topic] += 1

        # perform collapsed Gibbs sampling
        iter_num = 0
        while iter_num < self.max_iter:
            for doc_idx, document in enumerate(documents):
                for word_idx, word in enumerate(document):

                    # decrement counts
                    previous_topic = self.doc2word_idx2topic[doc_idx][word_idx]
                    self.doc2topic2cnt[doc_idx][previous_topic] -= 1
                    self.topic2cnt[previous_topic] -= 1
                    self.word2topic2cnt[word][previous_topic] -= 1

                    # assign word to new topic
                    topic_masses = np.array([self.calculate_mass(self.doc2topic2cnt[doc_idx],
                                                                 self.word2topic2cnt[word][topic_idx],
                                                                 self.topic2cnt[topic_idx], topic_idx, word)
                                             for topic_idx in range(self.num_topics)])
                    topic_masses_norm = topic_masses / np.sum(topic_masses)

                    new_topic = np.random.choice(self.num_topics, 1, p=topic_masses_norm)

                    # update counts needed for updates
                    self.doc2topic2cnt[doc_idx][new_topic] += 1
                    self.topic2cnt[new_topic] += 1
                    self.word2topic2cnt[word][new_topic] += 1

            iter_num += 1

    def get_topic_distributions(self):
        vocab_size = len(self.word2topic2cnt)
        matrix = np.zeros((self.num_topics, vocab_size))
        for topic_idx in range(self.num_topics):
            for word_idx in range(vocab_size):
                matrix[topic_idx, word_idx] = ((self.word2topic2cnt[word_idx][topic_idx] + self.beta[word_idx]) /
                                               (self.topic2cnt[topic_idx]) + vocab_size * self.beta[word_idx])

        return matrix

    def get_document_distributions(self):
        matrix = np.zeros((self.num_docs, self.num_topics))
        for doc_idx in range(self.num_docs):
            topic2cnt = self.doc2topic2cnt[doc_idx]
            topic_assigned_num = sum([topic2cnt[topic_idx] > 0 for topic_idx in range(self.num_topics)])
            for topic_idx in range(self.num_topics):
                matrix[doc_idx, topic_idx] = ((self.doc2topic2cnt[doc_idx][topic_idx] + self.alpha[topic_idx]) /
                                              (topic_assigned_num + self.num_topics * self.alpha[topic_idx]))

        return matrix
