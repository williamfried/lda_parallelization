import numpy as np
import matplotlib.pyplot as plt


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
        self.doc2word2cnt = None

        np.random.seed(random_state)

    def calculate_mass(self, n_doc, n_word, n_all):
        '''Compute the unnormalized mass associated with a specific topic given the relevant count information.'''
        return (n_doc + self.alpha) * (n_word + self.beta) / (n_all + self.beta_sum)

    def compute_perplexity(self):
        '''Calculate perplexity score after each iteration to determine if the algorithm has converged. '''
        pass

    def update_counts(self, doc_id, word, topic, direction):
        val = 1 if direction == 'up' else -1
        self.doc2word2topic2cnt[doc_id][word][topic] += val
        self.doc2topic2cnt[doc_id][topic] += val
        self.topic2cnt[topic] += val
        self.word2topic2cnt[word][topic] += val

    def fit(self, doc2word2cnt):
        '''Perform LDA inference algorithm as described in paper.'''

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

                for random_topic in random_topics:
                    self.update_counts(doc_id, word, random_topic, 'up')

        self.vocab_size = len(self.word2topic2cnt)
        self.beta_sum = self.vocab_size * self.beta

        # perform collapsed Gibbs sampling
        iter_num = 0
        averages_cohs = []
        averages_cohs2 = []
        averages_cohs3 = []
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

            coherences = self.get_topic_coherence(30)
            averages_cohs.append(np.mean(coherences))
            coherences2 = self.get_topic_coherence2(30)
            averages_cohs2.append(np.mean(coherences2))
            coherences3 = self.get_topic_coherence3(30)
            averages_cohs3.append(np.mean(coherences3))

            iter_num += 1

        plt.figure()
        plt.plot(range(self.max_iter), averages_cohs, color='r', label='original way')
        plt.plot(range(self.max_iter), averages_cohs2, color='b', label='alternative way')
        plt.plot(range(self.max_iter), averages_cohs3, color='g', label='combined way')
        plt.xlabel('Number of iteration')
        plt.ylabel('Average topic coherence')
        plt.legend()
        plt.savefig('CoherenceConvergence.png')
        plt.show()

        # print(self.topic2cnt)

    def get_topic_distributions(self):
        '''Calculate word distribution for each topic using methodology described here:
        https://stats.stackexchange.com/questions/346329/in-lda-after-collapsed-gibbs-sampling-how-to-estimate-values-of-other-latent-v'''
        matrix = np.zeros((self.num_topics, self.vocab_size))
        for topic in range(self.num_topics):
            for word in range(self.vocab_size):
                matrix[topic, word] = ((self.word2topic2cnt[word][topic] + self.beta) /
                                       (self.topic2cnt[topic]) + self.beta_sum)

        return matrix

    def get_document_distributions(self):
        '''Calculation topic distribution for each document based on same source.'''
        matrix = np.zeros((self.num_docs, self.num_topics))
        for doc_id in range(self.num_docs):
            topic2cnt = self.doc2topic2cnt[doc_id]
            topic_assigned_num = sum([topic2cnt[topic] > 0 for topic in range(self.num_topics)])
            for topic in range(self.num_topics):
                matrix[doc_id, topic] = ((self.doc2topic2cnt[doc_id][topic] + self.alpha) /
                                         (topic_assigned_num + self.num_topics * self.alpha))

        return matrix

    def get_topic_coherence(self, top_word_num=40):

        # fist map each word to all the doc ids it appears in

        word2docs = {word: {doc for doc in range(self.num_docs) if word in self.doc2word2cnt[doc]}
                     for word in range(self.vocab_size)}

        coherences = []
        topic_distributions = self.get_topic_distributions()
        topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)
        top_words = topic_distributions_sorted[:, :top_word_num]

        top_words_all_topics = top_words.flatten() #

        word_tracker = set() #
        v_m_count_lst = [] #
        for topic in range(self.num_topics):
            top_words_topic = top_words[topic, :]
            coherence = 0

            for m, v_m in enumerate(top_words_topic[1:]):
                v_m_count = np.count_nonzero(top_words_all_topics == v_m)
                if v_m not in word_tracker: #
                    word_tracker.add(v_m) #
                    v_m_count_lst.append(v_m_count) #
                for v_l in top_words_topic[:m+1]:
                    v_l_count = np.count_nonzero(top_words_all_topics == v_l)
                    # get co-occurence
                    num = len(word2docs[v_m].intersection(word2docs[v_l])) + 1
                    denom = len(word2docs[v_l]) * v_m_count * v_l_count #
                    coherence += np.log(num / denom)

            coherences.append(coherence)

        # normalize by number of top words
        coherences = [k / top_word_num for k in coherences]
        print('avg topics:', np.mean(v_m_count_lst))
        print('Average topic coherence is : {}'.format(np.mean(coherences)))
        # print('With min and max respectively : {} & {}'.format(np.min(coherences), np.max(coherences)))

        return coherences

    def get_topic_coherence2(self, top_word_num=40):

        # fist map each word to all the doc ids it appears in

        word2docs = {word: {doc for doc in range(self.num_docs) if word in self.doc2word2cnt[doc]}
                     for word in range(self.vocab_size)}

        coherences = []
        topic_distributions = self.get_topic_distributions()
        topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)
        top_words = topic_distributions_sorted[:, :top_word_num]

        top_words_all_topics = top_words.flatten() #

        word_tracker = set() #
        v_m_count_lst = [] #
        for topic in range(self.num_topics):
            top_words_topic = top_words[topic, :]
            coherence = 0

            for m, v_m in enumerate(top_words_topic[1:]):
                v_m_count = np.count_nonzero(top_words_all_topics == v_m)
                if v_m not in word_tracker: #
                    word_tracker.add(v_m) #
                    v_m_count_lst.append(v_m_count) #
                for v_l in top_words_topic[:m+1]:
                    v_l_count = np.count_nonzero(top_words_all_topics == v_l)
                    # get co-occurence
                    num = len(word2docs[v_m].intersection(word2docs[v_l])) + 1
                    denom = len(word2docs[v_m]) * v_m_count * v_l_count #
                    coherence += np.log(num / denom)

            coherences.append(coherence)

        # normalize by number of top words
        coherences = [k / top_word_num for k in coherences]
        print('avg topics:', np.mean(v_m_count_lst))
        print('Average topic coherence is : {}'.format(np.mean(coherences)))
        # print('With min and max respectively : {} & {}'.format(np.min(coherences), np.max(coherences)))

        return coherences

    def get_topic_coherence3(self, top_word_num=40):

        # fist map each word to all the doc ids it appears in

        word2docs = {word: {doc for doc in range(self.num_docs) if word in self.doc2word2cnt[doc]}
                     for word in range(self.vocab_size)}

        coherences = []
        topic_distributions = self.get_topic_distributions()
        topic_distributions_sorted = np.argsort(-topic_distributions, axis=1)
        top_words = topic_distributions_sorted[:, :top_word_num]

        top_words_all_topics = top_words.flatten() #

        word_tracker = set() #
        v_m_count_lst = [] #
        for topic in range(self.num_topics):
            top_words_topic = top_words[topic, :]
            coherence = 0

            for m, v_m in enumerate(top_words_topic[1:]):
                v_m_count = np.count_nonzero(top_words_all_topics == v_m)
                if v_m not in word_tracker: #
                    word_tracker.add(v_m) #
                    v_m_count_lst.append(v_m_count) #
                for v_l in top_words_topic[:m+1]:
                    v_l_count = np.count_nonzero(top_words_all_topics == v_l)
                    # get co-occurence
                    num = len(word2docs[v_m].intersection(word2docs[v_l])) + 1
                    denom = (len(word2docs[v_m]) + len(word2docs[v_l])) * v_m_count * v_l_count #
                    coherence += np.log(num / denom)

            coherences.append(coherence)

        # normalize by number of top words
        coherences = [k / top_word_num for k in coherences]
        print('avg topics:', np.mean(v_m_count_lst))
        print('Average topic coherence is : {}'.format(np.mean(coherences)))
        # print('With min and max respectively : {} & {}'.format(np.min(coherences), np.max(coherences)))

        return coherences