from collections import defaultdict
from mpi4py import MPI
import numpy as np
import sys


class LDA:

    def __init__(self, num_topics, alpha=None, beta=None, batch_fraction=0.25, min_batch_size=100,
                 max_iter=50, random_state=205):
        self.num_topics = num_topics
        self.alpha = alpha if alpha else 1 / num_topics
        self.beta = beta if beta else 1 / num_topics
        self.max_iter = max_iter

        self.vocab_size = None
        self.beta_sum = None

        self.topic2cnt = None
        # self.word2topic2cnt = None
        self.doc2topic2cnt = None
        self.num_docs = None
        self.doc2word2topic2cnt = None
        self.word2docs = None
        self.token_queue = None

        np.random.seed(random_state)

        self.batch_fraction = batch_fraction
        self.batch_size = None
        self.min_batch_size = min_batch_size

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
        # self.word2topic2cnt[word][topic] += val

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

    def initialize(self, entire_doc2word2cnt):

        # partition documents into equally-sized chunks
        doc_partitions = self.elements_per_process(len(entire_doc2word2cnt))

        # assign chunk of documents to each node
        doc_ids = list(entire_doc2word2cnt.keys())
        docs_before = sum(doc_partitions[:self.mpi_rank])
        assigned_doc_ids = doc_ids[docs_before: docs_before + doc_partitions[self.mpi_rank]]
        doc2word2cnt = {i: entire_doc2word2cnt[doc_id] for i, doc_id in enumerate(assigned_doc_ids)}
        self.num_docs = len(doc2word2cnt)

        # number of words assigned to topic z across all documents
        self.topic2cnt = {i: 0 for i in range(self.num_topics)}

        # number of times each word is assigned to each topic across all documents
        word2topic2cnt = {}

        # number of words in each document that are assigned to each topic
        self.doc2topic2cnt = {doc_num: {i: 0 for i in range(self.num_topics)} for doc_num in range(self.num_docs)}

        # number of times each word is assigned to each topic in each document
        self.doc2word2topic2cnt = {i: {} for i in range(self.num_docs)}

        # keep track of documents in which each word appears
        self.word2docs = defaultdict(list)

        # randomly initialize word assignments
        for doc_id, word2cnt in doc2word2cnt.items():
            for word, word_cnt in word2cnt.items():

                self.doc2word2topic2cnt[doc_id][word] = {i: 0 for i in range(self.num_topics)}

                self.word2docs[word].append(doc_id)

                if word not in word2topic2cnt:
                    word2topic2cnt[word] = {i: 0 for i in range(self.num_topics)}

                # get random topics
                random_topics = np.random.choice(self.num_topics, word_cnt)

                for random_topic in random_topics:
                    self.update_counts(doc_id, word, random_topic, 'up')
                    word2topic2cnt[word][random_topic] += 1

        # each node sends topic count info to node 0
        topic2cnt_lst = self.comm.gather(self.topic2cnt, root=0)

        # node 0 combines (adds) the info from every other node to calculate complete topic count
        if self.mpi_rank == 0:
            topic2cnt_all_nodes = {i: 0 for i in range(self.num_topics)}
            for topic2cnt in topic2cnt_lst:
                for topic in topic2cnt:
                    topic2cnt_all_nodes[topic] += topic2cnt[topic]
        else:
            topic2cnt_all_nodes = None

        # node 0 sends complete topic count to each node
        self.topic2cnt = self.comm.bcast(topic2cnt_all_nodes, root=0)

        # each node sends word topic count info to node 0
        word2topic2cnt_lst = self.comm.gather(word2topic2cnt, root=0)

        # node 0 combines (adds) the info from every other node to calculate complete word topic count
        if self.mpi_rank == 0:
            word2topic2cnt_all_nodes = {}
            for word2topic2cnt in word2topic2cnt_lst:
                for word in word2topic2cnt:
                    if word not in word2topic2cnt_all_nodes:
                        word2topic2cnt_all_nodes[word] = {i: 0 for i in range(self.num_topics)}
                    topic2cnt = word2topic2cnt[word]
                    for topic in topic2cnt:
                        word2topic2cnt_all_nodes[word][topic] += topic2cnt[topic]

            vocab_size = len(word2topic2cnt_all_nodes)

            # node 0 splits the words in the vocabulary into equally-sized chunks to distribute to other nodes
            word_partitions = self.elements_per_process(vocab_size)
            scatter_lst = []
            words = list(word2topic2cnt_all_nodes.keys())
            start_idx = 0
            for word_partition in word_partitions:
                scatter_lst.append(([{word: word2topic2cnt_all_nodes[word]}
                                     for word in words[start_idx: start_idx + word_partition]], vocab_size))
                start_idx += word_partition
        else:
            scatter_lst = None

        # node 0 assigns a queue to each node
        self.token_queue, self.vocab_size = self.comm.scatter(scatter_lst, root=0)
        self.beta_sum = self.vocab_size * self.beta
        self.batch_size = round(len(self.token_queue) * self.batch_fraction)

    def fit(self, entire_doc2word2cnt):
        '''Perform LDA inference algorithm as described in paper.'''

        self.initialize(entire_doc2word2cnt)

        # perform collapsed Gibbs sampling
        iter_num = 0
        while iter_num < self.max_iter:
            print(iter_num)
            queue_len = len(self.token_queue)
            next_queue = []
            send_lst = []
            receive_lst = []
            req = self.comm.Irecv(receive_lst, source=(self.mpi_rank - 1) % self.mpi_size)

            # process each token in queue
            for i, token in enumerate(self.token_queue):
                
                if req.Test():
                    next_queue.extend(receive_lst)
                    req = self.comm.Irecv(receive_lst, source=(self.mpi_rank - 1) % self.mpi_size)

                word = list(token.keys())[0]
                w_vec = token[word]
                relevant_doc_ids = self.word2docs[word]

                for doc_id in relevant_doc_ids:
                    topic2cnt = self.doc2word2topic2cnt[doc_id][word]
                    previous_topics = []
                    for topic, cnt in topic2cnt.items():
                        if cnt > 0:
                            previous_topics.extend([topic] * cnt)

                    np.random.shuffle(previous_topics)

                    for previous_topic in previous_topics:

                        # decrement counts
                        self.update_counts(doc_id, word, previous_topic, 'down')
                        w_vec[previous_topic] -= 1

                        # assign word to new topic
                        topic_masses = np.array([self.calculate_mass(self.doc2topic2cnt[doc_id][topic_idx],
                                                                     w_vec[topic_idx],
                                                                     self.topic2cnt[topic_idx])
                                                 for topic_idx in range(self.num_topics)])
                        topic_masses_norm = topic_masses / np.sum(topic_masses)

                        new_topic = np.random.choice(self.num_topics, 1, p=topic_masses_norm)[0]

                        # increment counts
                        self.update_counts(doc_id, word, new_topic, 'up')
                        w_vec[new_topic] += 1

                send_lst.append(w_vec)
                if len(send_lst) >= self.batch_size and i < (queue_len - self.min_batch_size):
                    self.comm.send(send_lst, dest=(self.mpi_rank + 1) % self.mpi_size)
                    send_lst = []

            self.comm.send(send_lst, dest=(self.mpi_rank + 1) % self.mpi_size)
            send_lst = []
            self.token_queue = next_queue
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
