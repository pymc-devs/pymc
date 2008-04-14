from scipy.sparse import csc_matrix
import numpy as np



cdef extern from "string.h":
    # to, from, number of bytes
    void *memcpy(void*, void*, int)

cdef extern from "taucs.h":
    # ctypedef double taucs_double

    ctypedef union taucs_values:
        void* v
        double* d
        float* s

    ctypedef struct taucs_ccs_matrix:
        int n, m, flags, *colptr, *rowind
        taucs_values values

    cdef int TAUCS_SUCCESS, TAUCS_ERROR, TAUCS_ERROR_NOMEM, TAUCS_ERROR_BADARGS, TAUCS_ERROR_INDEFINITE, TAUCS_ERROR_MAXDEPTH, TAUCS_INT
    cdef int TAUCS_DOUBLE, TAUCS_SINGLE, TAUCS_DCOMPLEX, TAUCS_SCOMPLEX, TAUCS_LOWER, TAUCS_UPPER, TAUCS_TRIANGULAR, TAUCS_SYMMETRIC
    cdef int TAUCS_HERMITIAN, TAUCS_PATTERN, TAUCS_METHOD_LLT, TAUCS_METHOD_LDLT, TAUCS_METHOD_PLU, TAUCS_VARIANT_SNMF, TAUCS_VARIANT_SNLL

    void* taucs_ccs_factor_llt_mf(taucs_ccs_matrix*)
    int taucs_supernodal_solve_llt(void*, double*, double*)
    taucs_ccs_matrix* taucs_supernodal_factor_to_ccs(void*)
    
    
cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object)


cdef taucs_ccs_matrix* scipy_to_taucs(object CSC, char* type, char* uplo, int taucs_type) except *:
    # Converts scipy CSC matrices to taucs matrices

    cdef int n, m, flags, *colptr, *rowind
    cdef double *values
    cdef taucs_ccs_matrix taucs_out

    # Get pointer to data
    values = <double*> PyArray_DATA(CSC.data)

    # Get shape
    n = CSC.shape[1]
    m = CSC.shape[0]
    
    # Get pointers to index pointers and row indices.
    colptr = <int*> PyArray_DATA(CSC.indptr)
    rowind = <int*> PyArray_DATA(CSC.indices)


    # Do 'bitwise OR' flags... hopefully this works.
    flags = taucs_type

    # Triangular, symmetric or other
    if type == 'S':
        flags = flags | TAUCS_SYMMETRIC
    elif type == 'T':
        flags = flags | TAUCS_TRIANGULAR
        
    # Upper, lower or other
    if uplo=='U':
        flags = flags | TAUCS_UPPER
    if uplo=='L':
        flags = flags | TAUCS_LOWER
        
    # Assign members of taucs matrix struct    
    taucs_out.n = n
    taucs_out.m = m
    taucs_out.flags = flags
    taucs_out.colptr = colptr
    taucs_out.rowind = rowind
    taucs_out.values.d = values
    
    # Return pointer.
    return &taucs_out
    
    
    
cdef object taucs_to_scipy(taucs_ccs_matrix* taucs_in_ptr):
    # Converts taucs matrices to scipy CSC matrices.

    cdef object sparse_out, data, indptr, indices
    cdef taucs_ccs_matrix taucs_in
    
    cdef int npts
    cdef int np1
    cdef int i
    
    cdef double *values_in
    cdef int *colptr_in, *rowind_in
    
    taucs_in = taucs_in_ptr[0]
    
    # Get pointers to data, col ptrs, row indices from taucs matrix.
    values_in = taucs_in.values.d
    colptr_in = taucs_in.colptr
    rowind_in = taucs_in.rowind
    
    # Get total number of points.
    np1 = taucs_in.n+1
    
    # Get number of columns plus one.
    npts = taucs_in.colptr[taucs_in.n]    

    # Make components of sparse matrix.
    data = np.zeros(npts, dtype=float)
    indices = np.zeros(npts, dtype=int)
    indptr = np.zeros(np1, dtype=int)    

    # Copy into sparse matrix. Try to make this faster!
    for i from 0<=i<npts:
        data[i] = values_in[i]
        indices[i] = rowind_in[i]
        
    for i from 0<=i<np1:
        indptr[i] = colptr_in[i]
    
    # Create sparse matrix.
    sparse_out = csc_matrix((data, indices, indptr))
    
    return sparse_out
    
    
def taucs_factor(scipy_matrix, uplo='U'):
    cdef taucs_ccs_matrix* taucs_matrix
    cdef taucs_ccs_matrix* factored_taucs_matrix
    cdef void* opaque_factor

    taucs_matrix = scipy_to_taucs(scipy_matrix, 'S', uplo, TAUCS_DOUBLE)
    opaque_factor = taucs_ccs_factor_llt_mf(taucs_matrix)
    factored_taucs_matrix = taucs_supernodal_factor_to_ccs(opaque_factor)

    return taucs_to_scipy(taucs_matrix)
