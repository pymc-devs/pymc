cdef extern from "numpy/ndarrayobject.h":
    void* PyArray_DATA(object)

from linalg_utils import diag_call    
# diag_call(x,cov_fun)
from scipy.sparse import csr_matrix
from numpy import array, zeros, hstack, asmatrix, asarray

cdef object _spcall_distance(object raw_cov_fun, object distance_fun, object cov_params,  object dist_params, object x, double cutoff, double amp, double scale):
    cdef object indices, indptr, data, sprow, n_sp, x_single, spind
    cdef double *data_ptr, *sprow_ptr, *dist_row_ptr, *sp_dist_row_ptr
    cdef int i, n, n_sofar, j, ind_now, len_sofar, n_current, chunk_rows
            
    # Figure out n
    n = x.shape[0]
    
    # Chunk size
    chunk_rows = 20
    
    # Pointers to columns
    indptr = zeros(n+1,dtype=int)

    # Allocate space for singleton
    if len(x.shape)>1:
        x_single = zeros((1,x.shape[1]), dtype=float)
    else:
        x_single = zeros(1, dtype=float)        
            
    # rescale cutoff
    cutoff=cutoff*scale        
            
    # Initialize indices and data    
    n_sofar = 0
    len_sofar = n
    n_current = 0

    indices = zeros(n*chunk_rows, dtype=int)
    data = zeros(n*chunk_rows, dtype=float)
    data_ptr = <double*> PyArray_DATA(data)

    # Initialize dense row, sparse row, sparse column indices
    sp_dist_row = asmatrix(zeros((1,n), dtype=float))
    sprow = sp_dist_row.copy()
    sp_dist_row_ptr = <double*> PyArray_DATA(sp_dist_row)
    sprow_ptr = <double*> PyArray_DATA(sprow)
    spind = zeros(n, dtype=int)
    
    x=asmatrix(x)


    for i from 0 <= i < n:

        q=x[i:,:].copy()
        # dist_row = distance_fun(x[i,:], x[i:,:], symm=False, **dist_params)
        dist_row = distance_fun(q[0,:], q, symm=False, **dist_params)
        dist_row_ptr = <double*> PyArray_DATA(dist_row)

        # Thin this row
        ind_now = 0
        for j from 0 <= j < n-i:
            if dist_row_ptr[j] < cutoff:
                sp_dist_row_ptr[ind_now] = dist_row_ptr[j] / scale
                spind[ind_now] = j
                ind_now = ind_now + 1

        for j from 0 <= j < ind_now:
            sprow_ptr[j] = sp_dist_row_ptr[j]
        p=sprow[0,:ind_now]
        raw_cov_fun(p, **cov_params)

        sprow_ptr = <double*> PyArray_DATA(p)
        for j from 0 <= j < ind_now:
            sprow_ptr[j] = sprow_ptr[j] * amp

        # Add new row pointer.        
        n_new = ind_now
        indptr[i+1] = indptr[i] + n_new

        # Malloc new memory if needed.
        if n_sofar + n_new > len_sofar:
            indices = hstack((indices,zeros(n*chunk_rows, dtype=int)))
            data = hstack((data,zeros(n*chunk_rows, dtype=float)))
            len_sofar = len_sofar + n*chunk_rows

        # Add new data and column indices.
        n_current = n_sofar + n_new
        data[n_sofar:n_current] = p
        indices[n_sofar:n_current] = spind[:n_new]+i
        n_sofar = n_current

    # return
    return csr_matrix((data,indices,indptr),(n,n))
    
    
cdef object _spcall_value(object cov_fun, object params, object x, object diag, double cutoff):
    cdef object indices, indptr, data, row, sprow, n_sp, x_single, spind
    cdef double *data_ptr, *row_ptr, *sprow_ptr
    cdef int i, n, n_sofar, j, ind_now, len_sofar, n_current, chunk_rows
    
    # Figure out n
    n = x.shape[0]
    
    # Chunk size
    chunk_rows = 20
    
    # Pointers to columns
    indptr = zeros(n+1,dtype=int)

    # Allocate space for singleton
    if len(x.shape)>1:
        x_single = zeros((1,x.shape[1]), dtype=float)
    else:
        x_single = zeros(1, dtype=float)
    
    # Figure out maximum variance, scale cutoff.
    cutoff = cutoff * diag.max()
    
    # Initialize indices and data    
    n_sofar = 0
    len_sofar = n*chunk_rows
    n_current = 0
    
    indices = zeros(n*chunk_rows, dtype=int)
    data = zeros(n*chunk_rows, dtype=float)
    data_ptr = <double*> PyArray_DATA(data)
    
    # Initialize dense row, sparse row, sparse column indices
    sprow = zeros(n, dtype=float)
    sprow_ptr = <double*> PyArray_DATA(sprow)
    spind = zeros(n, dtype=int)
    
    for i from 0 <= i < n:

        # Evaluate this row
        x_single[:] = x[i,:]
        row = cov_fun(x_single, x[i:,:], **params)
        row_ptr = <double*> PyArray_DATA(row)
        
        # Thin this row
        ind_now = 0
        for j from 0 <= j < n-i:
            if row_ptr[j] > cutoff:
                sprow_ptr[ind_now] = row_ptr[j]
                spind[ind_now] = j
                ind_now = ind_now + 1

        # Add new row pointer.        
        n_new = ind_now
        indptr[i+1] = indptr[i] + n_new

        # Malloc new memory if needed.
        if n_sofar + n_new > len_sofar:
            indices = hstack((indices,zeros(n*chunk_rows, dtype=int)))
            data = hstack((data,zeros(n*chunk_rows, dtype=float)))
            len_sofar = len_sofar + n*chunk_rows

        # Add new data and column indices.
        n_current = n_sofar + n_new
        data[n_sofar:n_current] = sprow[:n_new]
        indices[n_sofar:n_current] = spind[:n_new]+i
        n_sofar = n_current
        
    # return
    return csr_matrix((data,indices,indptr),(n,n))
    
def spcall_value(cov_fun, params, x, diag, cutoff):
    """
    spcall_value(cov_fun, params, x, diag, cutoff)
    
    - cov_fun 
        A covariance function.        
    - params 
        A dictionary.
    - x 
        A regularized array of observation points.
    - cutoff 
        The relative drop value. Entries that are less than C.max()*cutoff
        will be zeroed.
    - diag 
        The diagonal of the covariance.
    """
    return _spcall_value(cov_fun, params, x, diag, cutoff)
    
def spcall_distance(cov_fun, params, x, cutoff):
    """
    spcall_distance(cov_fun, params, x, cutoff)
    
    - cov_fun 
        A covariance function.        
    - params 
        A dictionary.
    - x 
        A regularized array of observation points.
    - cutoff 
        The relative drop value. Entries that are further away than 
        scale*cutoff will be zeroed.   
    """
    # Extract distance and raw covariance function.
    distance_fun = cov_fun.distance_fun
    raw_cov_fun = cov_fun.raw_cov_fun
    
    # Deconstruct parameter dictionary
    amp = params['amp'] ** 2
    scale = params['scale']

    cov_params = {}
    if hasattr(cov_fun, 'extra_cov_params'):
        for key in cov_fun.extra_cov_params.iterkeys():
            cov_params[key] = params[key]
        
    dist_params = {}
    if hasattr(cov_fun, 'extra_distance_params'):
        for key in cov_fun.extra_distance_params.iterkeys():
            dist_params[key] = params[key]
    
    return _spcall_distance(raw_cov_fun, distance_fun, cov_params, dist_params, x, cutoff, amp, scale)