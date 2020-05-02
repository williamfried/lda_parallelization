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
from mpi4py import MPI  # Python import
from mpi4py cimport MPI  # types for above python import
from mpi4py cimport libmpi as mpi  # C API
import os.path
from libc.stdlib cimport malloc, calloc, free


# Indicates that LDA is not subclassed;
# so we can have python versions of nogil instance methods
@cython.final
cdef class LDA:
    # Consider representing the assignments as a linked sparse matrix (linked
    # in both row and column directions).
    cdef vector[vector[vector[int]]] *assignment_ptr
    cdef readonly int size_vocab, size_corpus, num_topics
    # Note these two could be represented as maps or unordered_maps
    cdef readonly int[:, ::1] n_token, n_doc
    cdef readonly int[::1] n_all
    cdef readonly double alpha, beta, beta_sum
    cdef readonly int num_threads
    # We only need access to bitgen_states_ptr, but we keep rand_gen
    # to make sure that they don't get garbage collected
    cdef list rand_gens
    cdef vector[bitgen_t *] *bitgen_states_ptr
    cdef vector[omp_lock_t] *omp_locks_ptr
    cdef mpi.MPI_Comm mpi_comm
    cdef int mpi_size
    cdef int mpi_rank
    cdef (int, int, int) current_tokens
    # Defined just to allocate the memory in the init;
    # would otherwise require lots of mallocs and frees of same size
    cdef double[:, ::1] pmf


    def __cinit__(self, dir_path, num_topics, alpha, beta, num_threads=1,
                  seed=205):
        cdef int i
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.num_threads = num_threads
        self.mpi_comm = mpi.MPI_COMM_WORLD
        mpi.MPI_Comm_size(self.mpi_comm, &self.mpi_size)
        mpi.MPI_Comm_rank(self.mpi_comm, &self.mpi_rank)
        seed_sequence = SeedSequence(seed).spawn(self.mpi_size)[self.mpi_rank]
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
        cdef int token, doc, count
        cdef str line
        cdef list line_split, occurrences
        cdef int[::1] temp_n_all
        cdef int[:, ::1] temp_n_token
        with open(
                os.path.join(dir_path, "part-{:05}".format(self.mpi_rank))
        ) as file:
            # Perhaps reading from the header is not a good idea
            # because it is not naturally output from Spark
            line_split = file.readline().strip().split(",")
            self.size_vocab = int(line_split[0])
            self.size_corpus = int(line_split[1])
            temp_n_token = np.ascontiguousarray(
                np.zeros((self.size_vocab, self.num_topics), dtype=np.intc)
            )
            self.n_doc = np.ascontiguousarray(
                np.zeros((self.size_corpus, self.num_topics), dtype=np.intc)
            )
            self.n_all = np.ascontiguousarray(
                np.zeros(self.num_topics, dtype=np.intc)
            )
            self.assignment_ptr = \
                new vector[vector[vector[int]]](self.size_vocab)
            for doc, line in enumerate(file):
                line_split = line.strip().split(",")
                iterline = map(int, iter(line_split[1:]))
                occurrences = [(i, next(iterline)) for i in iterline]
                for token, count in occurrences:
                    random_assignment = (
                        self.num_topics * np_rng.random(count)
                    ).astype(int)
                    for i in random_assignment:
                        temp_n_token[token, i] += 1
                        self.n_doc[doc, i] += 1
                        self.n_all[i] += 1
                    self.assignment_ptr[0][token].push_back(random_assignment)
                    self.assignment_ptr[0][token].back().push_back(doc)
        self.beta_sum = self.size_vocab * self.beta
        self.current_tokens = (
            self.size_vocab * self.mpi_rank // self.mpi_size,
            self.size_vocab * (self.mpi_rank + 1) // self.mpi_size,
            self.mpi_rank
        )
        # Might have an extra row
        self.n_token = np.ascontiguousarray(
            np.zeros(((self.size_vocab + self.mpi_size - 1) // self.mpi_size,
                      self.num_topics), dtype=np.intc)
        )
        mpi.MPI_Allreduce(mpi.MPI_IN_PLACE, <void *> &self.n_all[0],
                          self.num_topics, mpi.MPI_INT, mpi.MPI_SUM,
                          self.mpi_comm)
        cdef int[:, ::1] temp_m_view  # Not necessary, but cleaner
        for i in range(self.mpi_size):
            temp_m_view = temp_n_token[
                          (self.size_vocab * i // self.mpi_size)
                          :(self.size_vocab * (i + 1) // self.mpi_size)
            ]
            mpi.MPI_Reduce(
                <void *> &temp_m_view[0, 0],
                # Ignored unless i == self.mpi_rank
                <void *> &self.n_token[0, 0],
                temp_m_view.size, mpi.MPI_INT, mpi.MPI_SUM, i, self.mpi_comm
            )
        self.pmf = np.ascontiguousarray(
            np.empty((self.num_threads, self.num_topics), dtype=np.double)
        )
        self.omp_locks_ptr = new vector[omp_lock_t](self.num_topics)
        for i in range(self.num_topics):
            omp_init_lock(&(self.omp_locks_ptr[0][i]))


    cpdef void sample(self, int num_iterations=1) nogil:
        """Run num_iterations epochs of CGS"""
        cdef int token_index, token, doc_index, document, old_topic, new_topic
        cdef int i, j, new_split
        cdef int *n_all_upd = <int *> calloc(self.num_topics, sizeof(int))
        # Unfortunately still requires the gil
        # cdef int[::1] n_all_upd_mv = <int[:self.num_topics]> n_all_upd
        # Current iteration number is i // self.mpi_size
        for i in range(num_iterations * self.mpi_size):
            for token in prange(self.current_tokens[0], self.current_tokens[1],
                                nogil=True, num_threads=self.num_threads):
                token_index = token - self.current_tokens[0]
                for doc_index in range(self.assignment_ptr[0][token].size()):
                    document = self.assignment_ptr[0][token][doc_index].back()
                    for j in range(self.num_topics):
                        self.pmf[omp_get_thread_num(), j] = (
                            (self.n_doc[document, j] + self.alpha)
                            * (self.n_token[token_index, j] + self.beta)
                            / (self.n_all[j] + n_all_upd[j] + self.beta_sum)
                        )
                    for j in range(
                            self.assignment_ptr[0][token][doc_index].size() - 1
                    ):
                        old_topic = self.assignment_ptr[0][token][doc_index][j]
                        self.n_token[token_index, old_topic] -= 1
                        omp_set_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.n_doc[document, old_topic] -=1
                        n_all_upd[old_topic] -= 1
                        omp_unset_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.pmf[omp_get_thread_num(), old_topic] = (
                            (self.n_doc[document, old_topic] + self.alpha)
                            * (self.n_token[token_index, old_topic] + self.beta)
                            / (self.n_all[old_topic] + n_all_upd[old_topic]
                               + self.beta_sum)
                        )
                        new_topic = categorical_sample(
                            self.bitgen_states_ptr[0][omp_get_thread_num()],
                            self.pmf[omp_get_thread_num()],
                        )
                        self.assignment_ptr[0][token][doc_index][j] = new_topic
                        self.n_token[token_index, new_topic] += 1
                        omp_set_lock(&(self.omp_locks_ptr[0][new_topic]))
                        self.n_doc[document, new_topic] += 1
                        n_all_upd[new_topic] += 1
                        omp_unset_lock(&(self.omp_locks_ptr[0][new_topic]))
                        self.pmf[omp_get_thread_num(), new_topic] = (
                            (self.n_doc[document, new_topic] + self.alpha)
                            * (self.n_token[token_index, new_topic] + self.beta)
                            / (self.n_all[new_topic] + n_all_upd[new_topic]
                               + self.beta_sum)
                        )
            mpi.MPI_Allreduce(mpi.MPI_IN_PLACE, <void *> n_all_upd,
                              self.num_topics, mpi.MPI_INT, mpi.MPI_SUM,
                              self.mpi_comm)
            for j in range(self.num_topics):
                self.n_all[j] += n_all_upd[j]
                n_all_upd[j] = 0
            mpi.MPI_Sendrecv_replace(
                <void *> &self.n_token[0, 0],
                self.n_token.shape[0] * self.n_token.shape[1], mpi.MPI_INT,
                (self.mpi_rank + 1) % self.mpi_size, 0,
                (self.mpi_rank - 1) % self.mpi_size, 0,
                self.mpi_comm, mpi.MPI_STATUS_IGNORE
            )
            new_split = (self.current_tokens[2] + 1) % self.mpi_size
            self.current_tokens = (
                self.size_vocab * new_split // self.mpi_size,
                self.size_vocab * (new_split + 1) // self.mpi_size,
                new_split,
            )
        free(n_all_upd)

    def get_topic_distributions(self):
        """
        Posterior distribution over tokens for each topic such that
        output[i, j] = Pr[token j | topic i]
        """
        # output = np.asarray(self.n_token)
        # output = output + self.beta
        # output = (output / np.sum(output, axis=0)).T
        # return output
        pass

    def get_document_distributions(self):
        """
        Posterior distribution over topics for each document such that
        output[i, j] = Pr[topic j | document i]
        """
        # output = np.asarray(self.n_doc)
        # output = output + self.alpha
        # output = (output.T / np.sum(output, axis=1)).T
        # return output
        pass

    def __dealloc__(self):
        del self.assignment_ptr
        del self.bitgen_states_ptr
        cdef int i
        for i in range(self.num_topics):
            omp_destroy_lock(&(self.omp_locks_ptr[0][i]))
        del self.omp_locks_ptr
