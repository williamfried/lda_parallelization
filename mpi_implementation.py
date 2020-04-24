from collections import defaultdict
from mpi4py import MPI
import numpy as np
import sys
import time
from copy import deepcopy


class LDA:

    def __init__(self, num_topics, alpha=None, beta=None, batch_fraction=0.25, min_batch_size=100,
                 max_iter=10, random_state=207):
        self.num_topics = num_topics
        self.alpha = alpha if alpha else 1 / num_topics
        self.beta = beta if beta else 1 / num_topics

        self.vocab_size = None
        self.beta_sum = None

        # self.topic2cnt = None
        self.topic2cnt_local = None
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
        self.max_iter = max_iter * self.mpi_size
        self.s_token = 's_token'

        if self.mpi_rank == 0:
            self.topic2cnt_global = None
            self.doc2topic2cnt_global = None
            self.word2topic2cnt_global = None

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
        self.topic2cnt_local[topic] += val
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
        doc2word2cnt = {doc_id: entire_doc2word2cnt[doc_id] for doc_id in assigned_doc_ids}
        self.num_docs = len(doc2word2cnt)

        # number of words assigned to topic z across all documents
        self.topic2cnt_local = {i: 0 for i in range(self.num_topics)}

        # number of times each word is assigned to each topic across all documents
        word2topic2cnt = {}

        # number of words in each document that are assigned to each topic
        self.doc2topic2cnt = {doc_id: {i: 0 for i in range(self.num_topics)} for doc_id in assigned_doc_ids}

        # number of times each word is assigned to each topic in each document
        self.doc2word2topic2cnt = {doc_id: {} for doc_id in assigned_doc_ids}

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
        topic2cnt_lst = self.comm.gather(self.topic2cnt_local, root=0)

        # node 0 combines (adds) the info from every other node to calculate complete topic count
        if self.mpi_rank == 0:
            topic2cnt_all_nodes = {i: 0 for i in range(self.num_topics)}
            for topic2cnt in topic2cnt_lst:
                for topic in topic2cnt:
                    topic2cnt_all_nodes[topic] += topic2cnt[topic]
        else:
            topic2cnt_all_nodes = None

        # node 0 sends complete topic count to each node
        self.topic2cnt_local = self.comm.bcast(topic2cnt_all_nodes, root=0)

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

            # shuffle words so that each node receives random words (words with lower indices appear on average
            # more times in the corpus, so they are more expensive to process)
            np.random.shuffle(words)

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

        if self.mpi_rank == 0:
            self.token_queue.append({self.s_token: self.topic2cnt_local})

    def fit(self, entire_doc2word2cnt):
        '''Perform LDA inference algorithm as described in paper.'''

        self.initialize(entire_doc2word2cnt)

        iter_num = 0
        terminate = False
        next_queue = []
        send_queue = []

        topic2cnt_snapshot = self.topic2cnt_local

        max_words_per_iteration = len(self.token_queue) + 2

        while iter_num < self.max_iter:

            # print(f'node {self.mpi_rank} is on iteration {iter_num}')
            # sys.stdout.flush()

            if iter_num > 0:
                self.token_queue = next_queue

            next_queue = []
            # send_queue = []
            send_queue_temp = []

            num_received = len(self.token_queue)

            # process each token in queue
            # for i, token in enumerate(self.token_queue):
            # for i in range(max_words_per_iteration):
            i = 0
            while i < max_words_per_iteration:

                # if self.mpi_rank == 0:
                #     print(i)

                # if self.comm.Iprobe(source=(self.mpi_rank - 1) % self.mpi_size, tag=2):
                #     prev_queue = self.comm.recv(source=(self.mpi_rank - 1) % self.mpi_size, tag=2)
                #     print(i)
                #     print(len(prev_queue))
                #     # sys.stdout.flush()
                #     self.token_queue.extend(prev_queue)

                if self.comm.Iprobe(source=(self.mpi_rank - 1) % self.mpi_size, tag=1):
                    receive_lst = self.comm.recv(source=(self.mpi_rank - 1) % self.mpi_size, tag=1)
                    # if len(receive_lst) == 1:
                    #     self.token_queue.append(receive_lst)
                    # if num_received < max_words_per_iteration - 5:
                    #     self.token_queue.extend(receive_lst)
                    #     num_received += len(receive_lst)
                    # elif len(next_queue) > max_words_per_iteration - 100:
                    #     print(self.mpi_rank, 'svbegaeg')
                    #     sys.stdout.flush()
                    #     self.comm.send(receive_lst, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)
                    #     # next_next_queue.extend(receive_lst)
                    # else:
                    #     next_queue.extend(receive_lst)
                    next_queue.extend(receive_lst)
                        #print(f'{self.mpi_rank}: {len(next_queue)}')

                # if self.mpi_rank == 0:
                #     print(i, self.token_queue[i])

                try:
                    token = self.token_queue[i]
                    word = list(token.keys())[0]

                    if word == self.s_token:
                        topic2cnt_global = token[word]
                        for topic in topic2cnt_global:
                            topic2cnt_global[topic] += (self.topic2cnt_local[topic] - topic2cnt_snapshot[topic])
                        topic2cnt_snapshot = topic2cnt_global
                        self.topic2cnt_local = topic2cnt_global
                        # self.comm.send({word: topic2cnt_global}, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)
                        send_queue.append({word: topic2cnt_global})

                    else:
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
                                                                             self.topic2cnt_local[topic_idx])
                                                         for topic_idx in range(self.num_topics)])
                                topic_masses_norm = topic_masses / np.sum(topic_masses)

                                new_topic = np.random.choice(self.num_topics, 1, p=topic_masses_norm)[0]

                                # increment counts
                                self.update_counts(doc_id, word, new_topic, 'up')
                                w_vec[new_topic] += 1

                        send_queue.append({word: w_vec})

                    if len(send_queue) >= self.batch_size and i < (max_words_per_iteration - self.min_batch_size):
                        self.comm.send(send_queue, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)
                        send_queue = []

                    i += 1

                except IndexError:
                    if i > max_words_per_iteration - 5:
                        break

                    # if len(send_queue):
                    #     self.comm.send(send_queue, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)
                    #     send_queue = []

                if self.comm.Iprobe(source=(self.mpi_rank + 1) % self.mpi_size, tag=9):
                    # message = self.comm.recv(source=(self.mpi_rank + 1) % self.mpi_size, tag=9)
                    # if message == 'reset':
                    #     print('here again')
                    #     sys.stdout.flush()
                    #     self.comm.isend('reset', dest=(self.mpi_rank - 1) % self.mpi_size, tag=9)
                    #     terminate1 = True
                    #     break
                    # else:
                    self.comm.isend(None, dest=(self.mpi_rank - 1) % self.mpi_size, tag=9)
                    terminate = True
                    break

            if terminate:
                break

            print(f'node {self.mpi_rank} is on iteration {iter_num} and has {num_received} tokens')
            sys.stdout.flush()
            # if next_next_queue:
            #     print(f'node{self.mpi_rank} has next next queue of length {len(next_next_queue)}')

            #self.comm.isend(send_queue, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)

            # if self.comm.Iprobe(source=(self.mpi_rank - 1) % self.mpi_size, tag=1):
            #     receive_lst = self.comm.recv(source=(self.mpi_rank - 1) % self.mpi_size, tag=1)
            #     # if len(receive_lst) == 1:
            #     #     self.token_queue.append(receive_lst)
            #     if num_received < max_words_per_iteration - 5:
            #         print('option1')
            #         self.token_queue.extend(receive_lst)
            #         num_received += len(receive_lst)
            #     elif len(next_queue) > max_words_per_iteration - 100:
            #         print('option2')
            #         self.comm.send(receive_lst, dest=(self.mpi_rank + 1) % self.mpi_size, tag=1)
            #         # next_next_queue.extend(receive_lst)
            #     else:
            #         next_queue.extend(receive_lst)
            #         print('option3')
            #     sys.stdout.flush()

            print('send queue length:', len(send_queue))
            sys.stdout.flush()

            rec_send = self.comm.isend(send_queue, dest=(self.mpi_rank + 1) % self.mpi_size, tag=2)
            rec_recv = self.comm.irecv(200000, source=(self.mpi_rank - 1) % self.mpi_size, tag=2)
            rec_send.wait()
            receive_lst = rec_recv.wait()
            next_queue.extend(receive_lst)
            print('next queue length:', len(next_queue))



            self.comm.Barrier()

            #print(self.mpi_rank, 'past barrier')
           # time.sleep(0.1)
            iter_num += 1

        if not terminate:
            self.comm.isend(None, dest=(self.mpi_rank - 1) % self.mpi_size, tag=9)

        topic2cnt_lst = self.comm.gather(self.topic2cnt_local, root=0)
        doc2topic2cnt_lst = self.comm.gather(self.doc2topic2cnt, root=0)
        token_queue_lst = self.comm.gather(self.token_queue, root=0)
        next_queue_lst = self.comm.gather(next_queue, root=0)

        if self.mpi_rank == 0:
            topic2cnt_global = {i: 0 for i in range(self.num_topics)}
            for topic2cnt in topic2cnt_lst:
                for topic in topic2cnt:
                    topic2cnt_global[topic] += (topic2cnt[topic] / self.mpi_size)

            self.topic2cnt_global = {k: round(v) for k, v in topic2cnt_global.items()}

            self.doc2topic2cnt_global = {}
            for doc2topic2cnt in doc2topic2cnt_lst:
                self.doc2topic2cnt_global.update(doc2topic2cnt)

            self.word2topic2cnt_global = {}
            for token_queue in token_queue_lst:
                for token in token_queue:
                    self.word2topic2cnt_global.update(token)

            for next_queue in next_queue_lst:
                for token in next_queue:
                    self.word2topic2cnt_global.update(token)

            print(len(self.word2topic2cnt_global))

            del self.word2topic2cnt_global[self.s_token]

        else:
            sys.exit(0)

    def get_topic_distributions(self):
        '''Calculate word distribution for each topic using methodology described here:
        https://stats.stackexchange.com/questions/346329/in-lda-after-collapsed-gibbs-sampling-how-to-estimate-values-of-other-latent-v'''
        matrix = np.zeros((self.num_topics, self.vocab_size))
        for topic in range(self.num_topics):
            for word in range(self.vocab_size):
                matrix[topic, word] = ((self.word2topic2cnt_global[word][topic] + self.beta) /
                                       (self.topic2cnt_global[topic]) + self.beta_sum)

        return matrix

    def get_document_distributions(self):
        '''Calculation topic distribution for each document based on same source.'''
        matrix = np.zeros((self.num_docs, self.num_topics))
        for doc_id in range(self.num_docs):
            topic2cnt = self.doc2topic2cnt_global[doc_id]
            topic_assigned_num = sum([topic2cnt[topic] > 0 for topic in range(self.num_topics)])
            for topic in range(self.num_topics):
                matrix[doc_id, topic] = ((self.doc2topic2cnt_global[doc_id][topic] + self.alpha) /
                                         (topic_assigned_num + self.num_topics * self.alpha))

        return matrix
