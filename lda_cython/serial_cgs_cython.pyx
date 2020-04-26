# cython: language_level=3
# distutils: language = c++
cimport cython
import numpy as np
from libcpp.vector cimport vector
from sampler cimport categorical_sample
from numpy.random import PCG64, SeedSequence
from numpy.random cimport bitgen_t
from cpython.pycapsule cimport PyCapsule_GetPointer
from openmp cimport omp_lock_t, omp_get_num_threads, omp_get_thread_num, \
    omp_init_lock, omp_destroy_lock, omp_set_lock, omp_unset_lock
from cython.parallel cimport prange


# Indicates that LDA is not subclassed;
# so we can have python versions of nogil instance methods
@cython.final
cdef class LDA:
    cdef vector[vector[vector[int]]] *assignment_ptr
    cdef readonly int size_vocab, size_corpus, num_topics
    cdef readonly int[:, :] n_token, n_doc
    cdef readonly int[:] n_all
    cdef readonly double alpha, beta, beta_sum
    cdef readonly int num_threads
    # We only need access to bitgen_states_ptr, but we keep rand_gen
    # to make sure that they don't get garbage collected
    cdef object rand_gens
    cdef vector[bitgen_t *] *bitgen_states_ptr
    cdef vector[omp_lock_t] *omp_locks_ptr
    # Defined just to allocate the memory in the init;
    # would otherwise require lots of mallocs and frees of same size
    cdef double[:, :] pmf


    def __cinit__(self, file_name, num_topics, alpha, beta, num_threads=1,
                  seed=None):
        cdef int i, token, doc, count
        cdef str line
        cdef list header, line_split, occurrences
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.num_threads = num_threads
        seed_sequence = SeedSequence(seed if seed is not None else 205)
        seeds = seed_sequence.spawn(self.num_threads + 1)
        np_rng = np.random.default_rng(seeds[0])
        self.rand_gens = [PCG64(s) for s in seeds[1:self.num_threads + 1]]
        self.bitgen_states_ptr = new vector[bitgen_t *]()
        self.bitgen_states_ptr[0].reserve(self.num_threads)
        for i in range(self.num_threads):
            self.bitgen_states_ptr[0].push_back(
                <bitgen_t *> PyCapsule_GetPointer(
                    self.rand_gens[i].capsule, "BitGenerator"
                )
            )
        with open(file_name) as file:
            header = file.readline().strip().split(",")
            self.size_vocab = int(header[0])
            self.size_corpus = int(header[1])
            n_token_np = np.zeros((self.size_vocab, self.num_topics),
                                  dtype=np.intc)
            n_doc_np = np.zeros((self.size_corpus, self.num_topics),
                                dtype=np.intc)
            n_all_np = np.zeros(self.num_topics, dtype=np.intc)
            self.assignment_ptr = \
                new vector[vector[vector[int]]](self.size_vocab)
            for line in file:
                line_split = line.strip().split(",")
                token = int(line_split[0])
                iterline = map(int, iter(line_split[1:]))
                occurrences = [(i, next(iterline)) for i in iterline]
                self.assignment_ptr[0][token].reserve(len(occurrences))
                for doc, count in occurrences:
                    random_assignment = np_rng.choice(self.num_topics, count)
                    rand_topics, rand_counts = np.unique(random_assignment,
                                                         return_counts=True)
                    n_token_np[token, rand_topics] += rand_counts
                    n_doc_np[doc, rand_topics] += rand_counts
                    n_all_np[rand_topics] += rand_counts
                    self.assignment_ptr[0][token].push_back(random_assignment)
                    self.assignment_ptr[0][token].back().push_back(doc)
        self.beta_sum = self.size_vocab * self.beta
        self.n_token = n_token_np
        self.n_doc = n_doc_np
        self.n_all = n_all_np
        self.pmf = np.zeros((self.num_threads, self.num_topics),
                            dtype=np.double)
        self.omp_locks_ptr = new vector[omp_lock_t](self.num_topics)
        for i in range(self.num_topics):
            omp_init_lock(&(self.omp_locks_ptr[0][i]))


    cpdef void sample(self, int num_iterations=1) nogil:
        """Run num_iterations epochs of CGS"""
        cdef int token, doc_index, document, old_topic, new_topic, i, j
        for i in range(num_iterations):
            for token in prange(self.size_vocab, nogil=True,
                                num_threads=self.num_threads):
                for doc_index in range(self.assignment_ptr[0][token].size()):
                    document = self.assignment_ptr[0][token][doc_index].back()
                    for j in range(self.num_topics):
                        self.pmf[omp_get_thread_num(), j] = (
                            (self.n_doc[document, j] + self.alpha)
                            * (self.n_token[token, j] + self.beta)
                            / (self.n_all[j] + self.beta_sum)
                        )
                    for j in range(
                            self.assignment_ptr[0][token][doc_index].size() - 1
                    ):
                        old_topic = self.assignment_ptr[0][token][doc_index][j]
                        self.n_token[token, old_topic] -= 1
                        omp_set_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.n_doc[document, old_topic] -=1
                        self.n_all[old_topic] -= 1
                        omp_unset_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.pmf[omp_get_thread_num(), old_topic] = (
                            (self.n_doc[document, old_topic] + self.alpha)
                            * (self.n_token[token, old_topic] + self.beta)
                            / (self.n_all[old_topic] + self.beta_sum)
                        )
                        new_topic = categorical_sample(
                            self.bitgen_states_ptr[0][omp_get_thread_num()],
                            self.pmf[omp_get_thread_num()],
                        )
                        self.assignment_ptr[0][token][doc_index][j] = new_topic
                        self.n_token[token, new_topic] += 1
                        omp_set_lock(&(self.omp_locks_ptr[0][new_topic]))
                        self.n_doc[document, new_topic] += 1
                        self.n_all[new_topic] += 1
                        omp_unset_lock(&(self.omp_locks_ptr[0][new_topic]))
                        self.pmf[omp_get_thread_num(), new_topic] = (
                            (self.n_doc[document, new_topic] + self.alpha)
                            * (self.n_token[token, new_topic] + self.beta)
                            / (self.n_all[new_topic] + self.beta_sum)
                        )


    def get_topic_distributions(self):
        """
        Posterior distribution over tokens for each topic such that
        output[i, j] = Pr[token j | topic i]
        """
        output = np.asarray(self.n_token)
        output = output + self.beta
        output = (output / np.sum(output, axis=0)).T
        return output

    def get_document_distributions(self):
        """
        Posterior distribution over topics for each document such that
        output[i, j] = Pr[topic j | document i]
        """
        output = np.asarray(self.n_doc)
        output = output + self.alpha
        output = (output.T / np.sum(output, axis=1)).T
        return output

    def __dealloc__(self):
        del self.assignment_ptr
        del self.bitgen_states_ptr
        cdef int i
        for i in range(self.num_topics):
            omp_destroy_lock(&(self.omp_locks_ptr[0][i]))
        del self.omp_locks_ptr
