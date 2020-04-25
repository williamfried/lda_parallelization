from numpy.random cimport bitgen_t
cdef int categorical_sample(bitgen_t *bitgen_state, double[:] pmf,
                            double *cmf=*) nogil