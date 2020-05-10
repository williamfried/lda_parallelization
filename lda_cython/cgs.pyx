# cython: language_level=3
# distutils: language=c++
cimport cython
import numpy as np
from libcpp.vector cimport vector
from utils cimport categorical_sample, partial_merge_loc
from numpy.random import PCG64, SeedSequence
from numpy.random cimport bitgen_t
from cpython.pycapsule cimport PyCapsule_GetPointer
from openmp cimport omp_lock_t, omp_get_num_threads, omp_get_thread_num, \
    omp_init_lock, omp_destroy_lock, omp_set_lock, omp_unset_lock
from cython.parallel cimport prange
# from mpi4py import MPI  # Python import
# from mpi4py cimport MPI  # types for above python import
from mpi4py cimport libmpi as mpi  # C API
from libc.stdlib cimport malloc, calloc, free
import re
import fileinput


# Indicates that LDA is not subclassed;
# so we can have python versions of nogil instance methods
@cython.final
cdef class LDA:
    # Consider representing the assignments as a linked sparse matrix (linked
    # in both row and column directions).
    # {assignment_ptr[0][token][doc_index]} is a vector of length
    # occurrences + 1, where the last entry denotes the document number
    cdef vector[vector[vector[int]]] *assignment_ptr
    cdef readonly int size_vocab, size_corpus, num_topics
    # Note these two could be represented as maps or unordered_maps
    # {n_token[token_index][topic]} contains corresponding counts
    # Last row may or may not be used; the first
    # {self.current_tokens[1] - self.current_tokens[0]} rows are valid
    # Be careful! If constructed with shuffle, output is also shuffled.
    # Use {self.get_inverse_permutation} to recover original token_id.
    cdef readonly int[:, ::1] n_token
    # {(start, stop, split_num)} that denote the rows of n_token
    # such that {token == token_index + start}
    # Should always be the case that
    # {start == size_vocab * split_num // mpi_size} and that
    # {stop == size_vocab * (split_num + 1) // mpi_size}
    cdef readonly (int, int, int) current_tokens
    # {n_doc[doc][topic]} contains corresponding count
    cdef readonly int[:, ::1] n_doc
    cdef readonly int[::1] n_all
    cdef readonly double alpha, beta, beta_sum
    # Number of OMP threads to use
    cdef readonly int num_threads
    # We only need access to {bitgen_states_ptr}, but we keep rand_gen
    # to make sure that they don't get garbage collected
    cdef list rand_gens
    cdef vector[bitgen_t *] *bitgen_states_ptr
    # Randomly shuffle the token orders so that
    # {token == token_permutation[token_id_from_input]}
    cdef readonly int[::1] token_permutation
    # Vector of locks, each corresponding to a topic
    cdef vector[omp_lock_t] *omp_locks_ptr
    cdef mpi.MPI_Comm mpi_comm
    cdef readonly int mpi_size
    cdef readonly int mpi_rank
    # Defined just to allocate the memory;
    # would otherwise require lots of {malloc}s and {free}s of same size
    cdef double[:, ::1] pmf

    def __init__(self, file_list, num_topics, alpha, beta, size_vocab,
                 size_corpus, num_threads=1, seed=205, shuffle_words=False):
        """See {__cinit__}"""
        pass

    def __cinit__(self, file_list, num_topics, alpha, beta, size_vocab,
                  size_corpus, num_threads=1, seed=205, shuffle_words=False):
        cdef int i
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.size_vocab = size_vocab
        self.size_corpus = size_corpus
        self.num_threads = num_threads
        self.mpi_comm = mpi.MPI_COMM_WORLD
        mpi.MPI_Comm_size(self.mpi_comm, &self.mpi_size)
        mpi.MPI_Comm_rank(self.mpi_comm, &self.mpi_rank)
        self.beta_sum = self.size_vocab * self.beta
        self.current_tokens = (
            self.size_vocab * self.mpi_rank // self.mpi_size,
            self.size_vocab * (self.mpi_rank + 1) // self.mpi_size,
            self.mpi_rank
        )

        if not shuffle_words:
            self.token_permutation = np.arange(self.size_vocab, dtype=np.intc)
        else:
            # Unlike below, we want same randomness for each node
            np.random.seed(seed)
            self.token_permutation = np.random.permutation(
                self.size_vocab
            ).astype(np.intc)

        # Set random seed
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

        cdef int[:, ::1] temp_n_token
        temp_n_token = np.zeros((self.size_vocab, self.num_topics),
                                dtype=np.intc)
        # Last row may or may not be used; the first
        # {self.current_tokens[1] - self.current_tokens[0]} rows are valid
        self.n_token = np.zeros(((self.size_vocab + self.mpi_size - 1)
                                 // self.mpi_size, self.num_topics),
                                dtype=np.intc)
        self.n_doc = np.zeros((self.size_corpus, self.num_topics),
                              dtype=np.intc)
        self.n_all = np.zeros(self.num_topics, dtype=np.intc)
        self.assignment_ptr = \
            new vector[vector[vector[int]]](self.size_vocab)

        cdef int token, token_original, doc, count
        cdef str line
        cdef list line_split, occurrences
        with fileinput.input(file_list) as file:
            # Randomly initialize assignments and record assignments
            for doc, line in enumerate(file):
                line_split = re.findall(r"\d+", line)
                iterline = map(int, iter(line_split[1:]))
                # Not the most efficient because we already have iterator
                # but much more readable
                occurrences = [(i, next(iterline))
                               for i in iterline]
                # {token_original} is token_id from input file
                # {token} is token index as far as code is concerned
                for token_original, count in occurrences:
                    if token_original < self.size_vocab:
                        token = self.token_permutation[token_original]
                        random_assignment = (
                            self.num_topics * np_rng.random(count)
                        ).astype(int)
                        for i in random_assignment:
                            temp_n_token[token, i] += 1
                            self.n_doc[doc, i] += 1
                            self.n_all[i] += 1
                        self.assignment_ptr[0][token].push_back(
                            random_assignment
                        )
                        self.assignment_ptr[0][token].back().push_back(doc)
        # Add up the {self.n_all} from the random assignments across nodes
        mpi.MPI_Allreduce(mpi.MPI_IN_PLACE, <void *> &self.n_all[0],
                          self.num_topics, mpi.MPI_INT, mpi.MPI_SUM,
                          self.mpi_comm)
        cdef int[:, ::1] temp_m_view  # Not necessary, but cleaner
        # Add up the each split of {temp_n_token}
        # from the random assignments across nodes
        # and write the result to the corresponding node's {self.n_token}
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
        # Just allocating memory here so it doesn't need to be done later
        # may be not necessary; certainly not all that clean, but saves
        # repeated {malloc}s and {free}s
        self.pmf = np.empty((self.num_threads, self.num_topics),
                            dtype=np.double)
        # Initialize the OMP locks, one per topic
        self.omp_locks_ptr = new vector[omp_lock_t](self.num_topics)
        for i in range(self.num_topics):
            omp_init_lock(&(self.omp_locks_ptr[0][i]))

    cpdef void sample(self, int num_iterations=1) nogil:
        """Run num_iterations epochs of CGS"""
        cdef int token_index, token, doc_index, document, old_topic, new_topic
        cdef int i, j, new_split
        cdef int *n_all_upd = <int *> calloc(self.num_topics, sizeof(int))
        # Unfortunately the next line still requires the gil
        # cdef int[::1] n_all_upd_mv = <int[:self.num_topics]> n_all_upd
        # Current iteration number is i // self.mpi_size
        for i in range(num_iterations * self.mpi_size):
            # Only look at the tokens that we have possession over
            for token in prange(self.current_tokens[0], self.current_tokens[1],
                                nogil=True, num_threads=self.num_threads,
                                schedule="dynamic"):
                # {token_index} required to index into {self.n_token}
                token_index = token - self.current_tokens[0]
                for doc_index in range(self.assignment_ptr[0][token].size()):
                    document = self.assignment_ptr[0][token][doc_index].back()
                    for j in range(self.num_topics):
                        self.pmf[omp_get_thread_num(), j] = (
                            (self.n_doc[document, j] + self.alpha)
                            * (self.n_token[token_index, j] + self.beta)
                            / (self.n_all[j] + n_all_upd[j] + self.beta_sum)
                        )
                    # The last entry of
                    # {self.assignment_ptr[0][token][doc_index]} is the
                    # document number, not another topic
                    for j in range(
                            self.assignment_ptr[0][token][doc_index].size() - 1
                    ):
                        old_topic = self.assignment_ptr[0][token][doc_index][j]
                        self.n_token[token_index, old_topic] -= 1
                        omp_set_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.n_doc[document, old_topic] -= 1
                        n_all_upd[old_topic] -= 1
                        omp_unset_lock(&(self.omp_locks_ptr[0][old_topic]))
                        self.pmf[omp_get_thread_num(), old_topic] = (
                            (self.n_doc[document, old_topic] + self.alpha)
                            * (self.n_token[token_index, old_topic]
                               + self.beta)
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
                            * (self.n_token[token_index, new_topic]
                               + self.beta)
                            / (self.n_all[new_topic] + n_all_upd[new_topic]
                               + self.beta_sum)
                        )
            # Every node updates their {self.n_all} and resets {n_all_upd} to 0
            mpi.MPI_Allreduce(mpi.MPI_IN_PLACE, <void *> n_all_upd,
                              self.num_topics, mpi.MPI_INT, mpi.MPI_SUM,
                              self.mpi_comm)
            for j in range(self.num_topics):
                self.n_all[j] += n_all_upd[j]
                n_all_upd[j] = 0
            # Send the chunk of tokens to the next node
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
        Be careful! If constructed with shuffle, output is also shuffled.
        Use {self.get_inverse_permutation} to recover original token_id.
        """
        output = np.asarray(self.n_token)[:(self.current_tokens[1]
                                            - self.current_tokens[0])]
        output = output + self.beta
        cdef double[::1] sum_arr = np.sum(output, axis=0)
        mpi.MPI_Allreduce(mpi.MPI_IN_PLACE, <void *> &sum_arr[0],
                          self.num_topics, mpi.MPI_INT, mpi.MPI_SUM,
                          self.mpi_comm)
        output = (output / np.asarray(sum_arr)).T
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

    def get_topic_coherence(self, num_top_tokens=40):
        """
        Get topic coherence, inspired by Optimizing Semantic Coherence
        in Topic Models (Minmo et al.) but with adjustments.
        Only the master node returns a value, but must be run by all nodes
        """
        # Get top words of each topic
        top_words = self.get_top_tokens(num_top_tokens)[:, :, 1]
        token_appearances = {
            token.item() : {vec.back()
                            for vec in self.assignment_ptr[0][token]}
            for token in np.nditer(top_words)
        }
        token_top_appearances = dict(zip(*np.unique(top_words,
                                                    return_counts=True)))
        cdef int num_pairs = (num_top_tokens * (num_top_tokens - 1)) // 2
        cdef int[:, ::1] local_co_occur = np.empty(
            (self.num_topics, num_pairs), dtype=np.intc
        )
        # This may repeat entries, but much easier than reducing dicts.
        # Perhaps take advantage of {np.unique(top_words)}
        cdef int[:, ::1] local_occur = np.empty(
            (self.num_topics, num_top_tokens), dtype=np.intc
        )
        cdef int i, j, k, l
        for i in range(self.num_topics):
            j = 0
            for m in range(num_top_tokens):
                local_occur[i, m] = len(token_appearances[top_words[i, m]])
            # This indexing order is slightly hard to follow; kept because
            # of original source
            for m in range(1, num_top_tokens):
                for l in range(m):
                    local_co_occur[i, j] = len(
                        token_appearances[top_words[i, m]]
                        & token_appearances[top_words[i, l]]
                    )
                    j += 1
        if self.mpi_rank != 0:
            mpi.MPI_Reduce(&local_co_occur[0, 0], NULL,
                           self.num_topics * num_pairs, mpi.MPI_INT,
                           mpi.MPI_SUM, 0, self.mpi_comm)
            mpi.MPI_Reduce(&local_occur[0, 0], NULL,
                           self.num_topics * num_top_tokens, mpi.MPI_INT,
                           mpi.MPI_SUM, 0, self.mpi_comm)
        else:
            mpi.MPI_Reduce(mpi.MPI_IN_PLACE, &local_co_occur[0, 0],
                           self.num_topics * num_pairs, mpi.MPI_INT,
                           mpi.MPI_SUM, 0, self.mpi_comm)
            mpi.MPI_Reduce(mpi.MPI_IN_PLACE, &local_occur[0, 0],
                           self.num_topics * num_top_tokens, mpi.MPI_INT,
                           mpi.MPI_SUM, 0, self.mpi_comm)
            topic_coherence = np.log(np.asarray(local_co_occur) + 1)
            # Decrease the weight of words found across different topics
            # which are also likely to be common words
            for i in range(self.num_topics):
                j = 0
                for m in range(1, num_top_tokens):
                    for l in range(m):
                        topic_coherence -= np.log(
                            token_top_appearances[top_words[i, m]]
                            * token_top_appearances[top_words[i, l]]
                        )
                        j += 1
            topic_coherence = np.sum(
                topic_coherence, axis=1
            )
            # I don't understand why this is so unsymmetric
            topic_coherence -= np.sum(
                np.log(np.add(local_occur, 1))
                * np.arange(num_top_tokens - 1, -1, -1),
                axis=1
            )
            # Perhaps it should be this instead
            # topic_coherence -= np.sum(np.log(local_occur), axis=1)
            return topic_coherence


    def get_top_tokens(self, num_top_tokens):
        """
        Returns the most common tokens of each topic along with their counts.
        {global_top[i, j, 1]} is the {j}th most common token in topic {i}
        and {global_top[i, j, 0]} is the corresponding count.
        Be careful! If constructed with shuffle, output is also shuffled.
        Use {self.get_inverse_permutation} to recover original token_id.
        """
        if num_top_tokens * self.mpi_size > self.size_vocab:
            raise ValueError("Too many words; try get_topic_distributions")
        n_token_arr = np.asarray(self.n_token[:(self.current_tokens[1]
                                                - self.current_tokens[0])],
                                 dtype=np.intc)
        local_top_id = np.argsort(-n_token_arr, axis=0)[:num_top_tokens]
        local_top_val = n_token_arr[local_top_id,
                                    np.arange(n_token_arr.shape[1])]
        # This assignment can probably be optimized along with the assignment
        # of {local_top_val}
        cdef int[:, :, ::1] local_top = np.ascontiguousarray(np.stack(
            (local_top_val.T, local_top_id.T.astype(np.intc)), axis=-1
        ))
        cdef int[:, :, ::1] global_top = np.empty_like(local_top)
        cdef mpi.MPI_Datatype sort_index_type
        # Perhaps more easily understood as
        # mpi.MPI_Type_contiguous(num_words, mpi.MPI_2INT, &sort_index_type)
        mpi.MPI_Type_contiguous(num_top_tokens * 2, mpi.MPI_INT,
                                &sort_index_type)
        mpi.MPI_Type_commit(&sort_index_type)
        cdef mpi.MPI_Op Partial_merge_loc
        mpi.MPI_Op_create(<mpi.MPI_User_function *> partial_merge_loc, True,
                          &Partial_merge_loc)
        mpi.MPI_Allreduce(<void *> &local_top[0, 0, 0],
                          <void *> &global_top[0, 0, 0],
                          self.num_topics, sort_index_type,
                          Partial_merge_loc, self.mpi_comm)
        mpi.MPI_Type_free(&sort_index_type)
        mpi.MPI_Op_free(&Partial_merge_loc)
        return np.asarray(global_top)

    def get_inverse_permutation(self):
        inverse = np.empty_like(self.token_permutation)
        inverse[self.token_permutation] = np.arange(self.size_vocab)
        return inverse

    def __dealloc__(self):
        del self.assignment_ptr
        del self.bitgen_states_ptr
        cdef int i
        for i in range(self.num_topics):
            omp_destroy_lock(&(self.omp_locks_ptr[0][i]))
        del self.omp_locks_ptr
