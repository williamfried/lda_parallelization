from mpi4py import MPI
import numpy as np
import sys


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

        self.comm = MPI.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.size

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

    def elements_per_process(self, total_elements):
        element_nums = []
        element_num = total_elements
        process_num = self.mpi_size

        while process_num > 0:
            ratio = element_num / process_num
            if ratio.is_integer():
                element_nums.extend([int(ratio)] * process_num)
                break
            num_assigned = int(np.ceil(ratio))
            element_nums.append(num_assigned)
            element_num -= num_assigned
            process_num -= 1

        return element_nums

    def fit(self, entire_doc2word2cnt):
        '''Perform LDA inference algorithm as described in paper.'''

        doc_nums = self.elements_per_process(len(entire_doc2word2cnt))

        doc_ids = list(entire_doc2word2cnt.keys())

        docs_before = sum(doc_nums[:self.mpi_rank])
        assigned_doc_ids = doc_ids[docs_before: docs_before + doc_nums[self.mpi_rank]]

        doc2word2cnt = {i: entire_doc2word2cnt[doc_id] for i, doc_id in enumerate(assigned_doc_ids)}

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
                    self.update_counts(doc_id, word, random_topic, 'up')

       # print(self.mpi_rank, self.topic2cnt)

        topic2cnt_lst = self.comm.gather(self.topic2cnt, root=0)

        if self.mpi_rank == 0:
            topic2cnt_all_nodes = {i: 0 for i in range(self.num_topics)}
            for topic2cnt in topic2cnt_lst:
                for topic in topic2cnt:
                    topic2cnt_all_nodes[topic] += topic2cnt[topic]
        else:
            topic2cnt_all_nodes = None

        self.topic2cnt = self.comm.bcast(topic2cnt_all_nodes, root=0)

        #print(self.mpi_rank, self.topic2cnt)

        word2topic2cnt_lst = self.comm.gather(self.word2topic2cnt, root=0)

        if self.mpi_rank == 0:
            word2topic2cnt_all_nodes = {}
            for word2topic2cnt in word2topic2cnt_lst:
                for word in word2topic2cnt:
                    if word not in


        # if self.mpi_rank == 0:
        #     return 0
        sys.exit(2)


        # post MPI sending
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

            iter_num += 1

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
