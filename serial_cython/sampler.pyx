# distutils: language = c++
cimport cython
from cpython.pycapsule cimport PyCapsule_GetPointer
from numpy.random import PCG64, SeedSequence
from numpy.random cimport bitgen_t
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

# np.import_array()


cdef int bin_search(double *cmf, int l, int r, double val) nogil:
    cdef int middle
    if r - l == 1:
        return l if val < cmf[l] else r
    else:
        middle = (r + l) // 2
        if val < cmf[middle]:
            return bin_search(cmf, l, middle, val)
        else:
            return bin_search(cmf, middle, r, val)


cdef int categorical_sample(bitgen_t *bitgen_state, double[:] pmf,
                            double *cmf=NULL) nogil:
    # Note that this does in work in the special case where prob[i] == 0,
    # in which case the bin_search on the cmf does not return what we want
    # if the rand_uniform happens to be exactly the value of my_cmf[i]
    cdef int total_classes = pmf.shape[0]
    cdef double rand_uniform = bitgen_state.next_double(bitgen_state)
    cdef double *my_cmf
    if cmf is NULL:
        my_cmf = <double *> malloc(total_classes * sizeof(double))
    else:
        my_cmf = cmf
    cdef int i
    cdef double total_mass = 0.
    for i in range(total_classes):
        total_mass += pmf[i]
        my_cmf[i] = total_mass
    rand_uniform *= total_mass
    cdef int output = bin_search(my_cmf, 0, total_classes, rand_uniform)
    if cmf is NULL:
        free(my_cmf)
    return output

# rand_gen = PCG64(0)
# cdef char *capsule_name = "BitGenerator"
# cdef bitgen_t *bitgen_state = <bitgen_t *> PyCapsule_GetPointer(
#     rand_gen.capsule, capsule_name
# )

# cdef double[:] pmf = np.array([0.1, 0.2, 0.5, 1])
# cpdef my_func(int n, double[:] pmf):
#     cdef int i
#     return [categorical_sample(bitgen_state, pmf) for i in range(n)]


# cdef int num_proc = 4
# seed_seq = SeedSequence(205)
# rand_gen_list = [PCG64(s) for s in seed_seq.spawn(num_proc)]
# cdef char *capsule_name = "BitGenerator"
# cdef bitgen_t **bitgen_arr = <bitgen_t **> malloc(num_proc * sizeof(bitgen_t *))
# cdef int i
# for i in range(num_proc):
#     bitgen_arr[i] = <bitgen_t *> PyCapsule_GetPointer(rand_gen_list[i].capsule,
#                                                       capsule_name)
# cdef double[:] p = np.array([0.1, 0.2, 0.5, 1])
# cdef double[:] counts = np.zeros(4)
# for i in prange(num_proc, nogil=True):
#     counts[categorical_sample(bitgen_arr[i], p)] += 1
# for i in range(4):
#     print(counts[i])
