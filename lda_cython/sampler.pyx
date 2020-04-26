# cython: language_level=3
# distutils: language = c++
from numpy.random cimport bitgen_t
from libcpp.vector cimport vector


cdef int bin_search(vector[double] *cmf, int l, int r, double val) nogil:
    cdef int middle
    if r - l == 1:
        return l if val < cmf[0][l] else r
    else:
        middle = (r + l) // 2
        if val < cmf[0][middle]:
            return bin_search(cmf, l, middle, val)
        else:
            return bin_search(cmf, middle, r, val)


cdef int categorical_sample(bitgen_t *bitgen_state, double[:] pmf) nogil:
    # Note that this does in work in the special case where prob[i] == 0,
    # in which case the bin_search on the cmf does not return what we want
    # if the rand_uniform happens to be exactly the value of cmf[i]
    cdef int total_classes = pmf.shape[0]
    cdef double rand_uniform = bitgen_state.next_double(bitgen_state)
    cdef vector[double] *cmf_ptr = new vector[double](total_classes)
    cdef int i
    cdef double total_mass = 0.
    for i in range(total_classes):
        total_mass += pmf[i]
        cmf_ptr[0][i] = total_mass
    rand_uniform *= total_mass
    cdef int output = bin_search(cmf_ptr, 0, total_classes, rand_uniform)
    del cmf_ptr
    return output
