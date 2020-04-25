# distutils: language = c++
cimport cython
import numpy as np
from libcpp.vector cimport vector
from sampler cimport categorical_sample
from numpy.random import PCG64
from numpy.random cimport bitgen_t
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdlib cimport malloc, free


# Indicates that LDA is not subclassed;
# so we can have python versions of nogil instance methods
@cython.final
cdef class LDA:
    cdef readonly vector[vector[vector[int]]] assignment
    cdef readonly int size_vocab, size_corpus, num_topics
    cdef readonly int[:, :] n_token, n_doc
    cdef readonly int[:] n_all
    cdef readonly double alpha, beta, beta_sum
    # We only need access to bitgen_state, but we keep rand_gen to make sure
    # that it doesn't get garbage collected
    cdef object rand_gen
    cdef bitgen_t *bitgen_state
    # Defined just to allocate the memory in the init;
    # would otherwise require lots of mallocs and frees of same size
    cdef double[:] pmf

    def __cinit__(self, file_name, num_topics, alpha, beta, seed=None):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        cdef int i, token
        with open(file_name) as file:
            header = file.readline().strip().split(",")
            self.size_vocab = int(header[0])
            self.size_corpus = int(header[1])
            self.num_topics = int(header[2])
            n_token_np = np.zeros((self.size_vocab, self.num_topics),
                                  dtype=np.intc)
            n_doc_np = np.zeros((self.size_corpus, self.num_topics),
                                dtype=np.intc)
            n_all_np = np.zeros(self.num_topics, dtype=np.intc)
            assignment_ptr = new vector[vector[vector[int]]](self.size_vocab)
            self.assignment = assignment_ptr[0]
            for line in file:
                line = line.strip().split(",")
                token = int(line[0])
                iterline = map(int, iter(line[1:]))
                occurrences = [(i, next(iterline)) for i in iterline]
                self.assignment[token].reserve(len(occurrences))
                for doc, count in occurrences:
                    random_assignment = np.random.choice(self.num_topics, count)
                    rand_topics, rand_counts = np.unique(random_assignment,
                                                         return_counts=True)
                    n_token_np[token, rand_topics] += rand_counts
                    n_doc_np[doc, rand_topics] += rand_counts
                    n_all_np[rand_topics] += rand_counts
                    self.assignment[token].push_back(random_assignment)
                    self.assignment[token].back().push_back(doc)
        self.beta_sum = self.size_vocab * beta
        self.n_token = n_token_np
        self.n_doc = n_doc_np
        self.n_all = n_all_np
        self.pmf = np.zeros(num_topics, dtype=np.double)
        self.rand_gen = PCG64(seed if seed is not None else 205)
        self.bitgen_state = <bitgen_t *> PyCapsule_GetPointer(
            self.rand_gen.capsule, "BitGenerator"
        )

    cpdef void sample(self, int num_iterations=1) nogil:
        """Run num_iterations epochs of CGS"""
        cdef int token, doc_index, document, old_topic, new_topic, i, j
        cdef double *cmf = <double *> malloc(self.num_topics * sizeof(double))
        for i in range(num_iterations):
            for token in range(self.size_vocab):
                for doc_index in range(self.assignment[token].size()):
                    document = self.assignment[token][doc_index].back()
                    for j in range(self.num_topics):
                        self.pmf[j] = (
                            (self.n_doc[document, j] + self.alpha)
                            * (self.n_token[token, j] + self.beta)
                            / (self.n_all[j] + self.beta_sum)
                        )
                    for j in range(
                            self.assignment[token][doc_index].size() - 1
                    ):
                        old_topic = self.assignment[token][doc_index][j]
                        self.n_doc[document, old_topic] -=1
                        self.n_token[token, old_topic] -= 1
                        self.n_all[old_topic] -= 1
                        self.pmf[old_topic] = (
                            (self.n_doc[document, old_topic] + self.alpha)
                            * (self.n_token[token, old_topic] + self.beta)
                            / (self.n_all[old_topic] + self.beta_sum)
                        )
                        new_topic = categorical_sample(
                            self.bitgen_state, self.pmf, cmf
                        )
                        self.assignment[token][doc_index][j] = new_topic
                        self.n_doc[document, new_topic] += 1
                        self.n_token[token, new_topic] += 1
                        self.n_all[new_topic] += 1
                        self.pmf[new_topic] = (
                            (self.n_doc[document, new_topic] + self.alpha)
                            * (self.n_token[token, new_topic] + self.beta)
                            / (self.n_all[new_topic] + self.beta_sum)
                        )
        free(cmf)

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
