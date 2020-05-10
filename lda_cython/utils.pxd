# cython: language_level=3
from numpy.random cimport bitgen_t
from mpi4py cimport libmpi as mpi
cdef int categorical_sample(bitgen_t *bitgen_state, double[:] pmf) nogil
cdef void partial_merge_loc(int *invec, int *inoutvec, int *len,
                            mpi.MPI_Datatype *datatype) nogil
