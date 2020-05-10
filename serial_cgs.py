import numpy as np


class LDA:

    def __init__(self, num_topics, alpha=None, beta=None, max_iter=50, tolerance=0.5, random_state=205):
        self.num_topics = num_topics
        self.alpha = alpha if alpha else 1 / num_topics
        self.beta = beta if beta else 1 / num_topics
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.vocab_size = None
        self.beta_sum = None

        self.topic2cnt = None
        self.word2topic2cnt = None
        self.doc2topic2cnt = None
        self.num_docs = None
        self.doc2word2topic2cnt = None
        self.doc2word2cnt = None

        np.random.seed(random_state)

    def calculate_mass(self, n_doc, n_word, n_all):
        '''Compute the unnormalized mass associated with a specific topic given the relevant count information.'''
        return (n_doc + self.alpha) * (n_word + self.beta) / (n_all + self.beta_sum)

    def update_counts(self, doc_id, word, topic, direction):
        '''Update document-assignment, word-assignment and topic-assignment counts.'''
        val = 1 if direction == 'up' else -1
        self.doc2word2topic2cnt[doc_id][word][topic] += val
        self.doc2topic2cnt[doc_id][topic] += val
        self.topic2cnt[topic] += val
        self.word2topic2cnt[word][topic] += val

    def fit(self, doc2word2cnt):
        '''Perform serial LDA inference algorithm.'''

        self.num_docs = len(doc2word2cnt)
        self.doc2word2cnt = doc2word2cnt

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

                #  update assignment counts
                for random_topic in random_topics:
                    self.update_counts(doc_id, word, random_topic, 'up')

        self.vocab_size = len(self.word2topic2cnt)
        self.beta_sum = self.vocab_size * self.beta

        # perform collapsed Gibbs sampling
        iter_num = 0

        # initialize topic coherence metric
        prev_topic_coherence = float('-inf')

        while iter_num < self.max_iter:
            for doc_id, word2cnt in doc2word2cnt.items():
                for word, word_cnt in word2cnt.items():

                    # get topic for each occurrence of word in document
                    topic2cnt = self.doc2word2topic2cnt[doc_id][word]
                    previous_topics = []
                    for topic, cnt in topic2cnt.items():
                        if cnt > 0:
                            previous_topics.extend([topic] * cnt)

                    np.random.shuffle(previous_topics)

                    for previous_topic in previous_topics:

                        # decrement counts
                        self.update_counts(doc_id, word, previous_topic, 'down')

                        # assign word to new topic
                        topic_masses = np.array([self.calculate_mass(self.doc2topic2cnt[doc_id][topic_idx],
                                                                     self.word2topic2cnt[word][topic_idx],
                                                                     self.topic2cnt[topic_idx])
                                                 for topic_idx in range(self.num_topics)])
                        topic_masses_norm = topic_masses / np.sum(topic_masses)

                        new_topic = np.random.choice(self.num_topics, 1, p=topic_masses_norm)[0]

                        # increment counts
                        self.update_counts(doc_id, word, new_topic, 'up')

            # check if sampler has converged
            topic_coherence = self.get_topic_coherence(30)
            if topic_coherence - prev_topic_coherence < self.tolerance:
                print('iterations to convergence:', iter_num + 1)
                break

            prev_topic_coherence = topic_coherence
            iter_num += 1

    def get_topic_distributions(self):
        '''Calculate word distribution for each topic.'''
        matrix = np.zeros((self.num_topics, self.vocab_size))
        for topic in range(self.num_topics):
            for word in range(self.vocab_size):
                matrix[topic, word] = ((self.word2topic2cnt[word][topic] + self.beta) /
                                       (self.topic2cnt[topic]) + self.beta_sum)

        return matrix

    def get_document_distributions(self):
        '''Calculation topic distribution for each document.'''
        matrix = np.zeros((self.num_docs, self.num_topics))
        for doc_id in range(self.num_docs):
            topic2cnt = self.doc2topic2cnt[doc_id]
            topic_assigned_num = sum([topic2cnt[topic] > 0 for topic in range(self.num_topics)])
            for topic in range(self.num_topics):
                matrix[doc_id, topic] = ((self.doc2topic2cnt[doc_id][topic] + self.alpha) /
                                         (topic_assigned_num + self.num_topics * self.alpha))

        return matrix

    def get_topic_coherence(self, top_word_num=40):

        # map each word to the documents in which it appears
        word2docs = {word: {doc for doc in range(self.num_docs) if word in self.doc2word2cnt[doc]}
                     for word in range(self.vocab_size)}

        coherences = []

        # calculate word distribution for each topic and get the most common words for each topic
        topic_distributions = self.get_topic_distributions()
        topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)
        top_words = topic_distributions_sorted[:, :top_word_num]

        top_words_all_topics = top_words.flatten()

        word_tracker = set()
        v_m_count_lst = []
        for topic in range(self.num_topics):
            top_words_topic = top_words[topic, :]
            coherence = 0

            # calculate topic coherence across all pairs of top words
            for m, v_m in enumerate(top_words_topic[1:]):
                v_m_count = np.count_nonzero(top_words_all_topics == v_m)
                if v_m not in word_tracker:
                    word_tracker.add(v_m)
                    v_m_count_lst.append(v_m_count)
                for v_l in top_words_topic[:m+1]:
                    v_l_count = np.count_nonzero(top_words_all_topics == v_l)

                    # number of documents in which the two words co-appear
                    num = len(word2docs[v_m].intersection(word2docs[v_l])) + 1
                    denom = (len(word2docs[v_m]) + len(word2docs[v_l])) * v_m_count * v_l_count
                    coherence += np.log(num / denom)

            coherences.append(coherence)

        # normalize by number of top words
        coherences = [k / top_word_num for k in coherences]

        # calculate average topic coherence of all topics
        return np.mean(coherences)
