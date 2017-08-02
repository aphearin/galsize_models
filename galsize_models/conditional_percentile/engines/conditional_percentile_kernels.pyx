"""
"""
from libc.math cimport fmod
cimport cython
import numpy as np


__all__ = ('conditional_percentile_kernel', )


cdef void insertion_update(long* arr, long n, long idx_in):
    cdef long i
    for i in range(n):
        if arr[i] >= idx_in:
            arr += 1


cdef void extraction_update(long* arr, long n, long idx_in):
    cdef long i
    for i in range(n):
        if arr[i] >= idx_in:
            arr -= 1


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef long bisect_left(double* arr, double value, long n):
    cdef long ifirst_subarr = 0
    cdef long ilast_subarr = n
    cdef long imid_subarr

    while ilast_subarr-ifirst_subarr >= 2:
        imid_subarr = (ifirst_subarr + ilast_subarr)/2
        if value > arr[imid_subarr]:
            ifirst_subarr = imid_subarr
        else:
            ilast_subarr = imid_subarr
    if value > arr[ifirst_subarr]:
        return ilast_subarr
    else:
        return ifirst_subarr



cdef void correspondence_indices_update(long* arr, long n, long idx_in, long idx_out):
    """
    """
    cdef long i
    cdef long increment = 0

    for i in range(1, n-1):
        arr[i] = arr[i-1]

    if idx_in < idx_out:
        for i in range(idx_in, idx_out):
            arr[i] += 1
    elif idx_in > idx_out + 1:
        for i in range(idx_out+1, idx_in):
            arr[i] -= 1

    arr[0] = idx_in
    if idx_in > idx_out:
        arr[0] -= 1


cdef void cython_insert_pop(double* arr, long idx_in, long idx_out, double value_in, long n):
    cdef int i

    if idx_in <= idx_out:
        for i in range(idx_out-1, idx_in-1, -1):
            arr[i+1] = arr[i]
    else:
        for i in range(idx_out, idx_in):
            arr[i] = arr[i+1]
    arr[idx_in] = value_in


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef void update_tables(double* cdf_value_table, long* correspondence_indices,
            double cdf_value_in, long n):
    """
    """
    cdef long idx_in = bisect_left(&cdf_value_table[0], cdf_value_in, n)
    cdef long idx_out = correspondence_indices[n-1]

    correspondence_indices_update(&correspondence_indices[0], n, idx_in, idx_out)

    cython_insert_pop(&cdf_value_table[0], idx_in, idx_out, cdf_value_in, n)



@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def conditional_percentile_kernel(double[:] initial_cdf_value_table,
        long[:] initial_correspondence_indices, double[:] values):
    """
    """
    cdef long num_values = values.shape[0]
    cdef long num_table = initial_cdf_value_table.shape[0]
    cdef long i
    cdef double[:] result = np.zeros(num_values, dtype='f8')

    return result








