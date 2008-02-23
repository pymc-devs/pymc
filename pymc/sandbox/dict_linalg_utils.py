from pymc import *
import numpy as np

class stochastic_keyed_dict_mat(object):
    """
    A sparse symmetric or upper-triangular matrix stored as a
    dictionary of dictionaries. Stochastics play the role of indices.
    """
    def __init__(self, diag, upper_tri, stochastics):
        self.diag = diag
        self.upper_tru = upper_tri
        self.stochastics = stochastics
        self.N_stochastics = len(stochastics)

def shape_preserving_sum(array_list):
    """
    Sums a list of arrays, forming an array.
    """
    out = array_list[0]
    for item in array_list[1:]:
        out += item
    return out

def backsolve_skdm(mat, b):
    """
    Backsolves upper-triangular stochastic-keyed dictionary matrix
    against a stochastic-keyed vector of values in-place.
    """
    pass
    
def backsolve_skdm_trans(mat, b):
    """
    Backsolves transpose of upper-triangular stochastic-keyed 
    dictionary matrix against a stochastic-keyed vector of values 
    in-place.
    """    
    pass
        
def sqrt_skdm(mat):
    """
    Takes a symmetric stochastic-keyed dictionary matrix
    (stored as upper triangle only) and converts it to a 
    Cholesky factor.
    """
    
    # dtrsm_wrap(a,b,side,transa,uplo)
    # dtrmm_wrap(a,b,side,transa,uplo)
    
    new_diag = {}
    new_upper_tri = {}
    for s in mat.stochastics:
        new_upper_tri[s] = []
        new_diag[s] = (False, [])
    
    for si in mat.stochastics:
        
        this_row = new_upper_tri[si]
        
        # Sum the contributions from above to the row.
        for elem in this_row.iteritems():
            
            # Start with original element.
            if mat.upper_tri.has_key(this_key):
                elem[1].append(-mat.upper_tri[this_key])
            
            # Sum contributions.
            if len(elem[1]) > 0:
                this_key = elem[0]
                this_row[this_key] = -shape_preserving_sim(elem[1])
            
            # If no contributions and nothing in the original matrix,
            # remove element.
            else:
                this_row.pop(elem[0])

        # If the diagonal element has still managed to be diagonal:
        if mat.diag[si][0] and len(new_diag[si][1])==0:
            new_diag[si][0] = True
            new_diag[si][1] = sqrt(mat.diag[si][1])
            this_diag = new_diag[si][1]

            # Backsolve diagonal against rest of row.
            for sj in this_row.iterkeys():
                this_row[sj] = (1./this_diag) * this_row[sj]
                    
        # If the diagonal element isn't diagonal:
        else:
            new_diag[si][0] = False
            
            # Append the original diagonal element.
            if mat.diag[si][0]:
                new_diag[si][1].append(np.diag(mat.diag[si][1]))
            else:
                new_diag[si][1].append(mat.diag[si][1])
            
            # Take Cholesky factor of original diagonal plus contributions.    
            new_diag[si][1] = np.linalg.cholesky(shape_preserving_sim(new_diag[si][1])).T
            this_diag = new_diag[si][1]

            for sj in this_row.iterkeys():
                this_row[sj] = dtrsm_wrap(this_diag,this_row[sj],'L','T','U')
                    
        # Append contributions to elements below and right.
        key_list = this_row.keys()
        for j in xrange(len(key_list)):

            key_j = key_list[j]

            # Append contributions to diagonal elements
            new_diag[key_j][1].append(np.dot(this_row[key_j].T,this_row[key_j]))

            # Append contributions to off-diagonal elements
            for k in xrange(len(key_list[j+1:])):

                key_k = key_list[k]
                # Append contributions to diagonal elements
                new_upper_tri[key_j][key_k].append(np.dot(this_row[key_j].T,this_row[key_k]))
                    
    return stochastic_keyed_dict_mat(new_diag, new_upper_tri, stochastics)
            
            
def square_skdm(mat):
    """
    Squares an upper-triangular stochastic-keyed dictionary matrixß.
    """
    
    old_diag = mat.diag
    old_upper_tri = mat.old_upper_tri
    stochastics = mat.stochastics
    
    new_upper_tri = {}
    new_diag = {}
    N_stochastics = len(stochastics)
    
    # Upper triangle
    for i in xrange(N_stochastics-1):
        new_upper_tri[i] = {}
        new_upper_tri_i_keys = old_upper_tri[i].iterkeys()
        for j in xrange(i+1,N_stochastics):
            new_upper_tri[i][j] = []

            for k in new_upper_tri_i_keys:
                if old_upper_tri[j].has_key(k):
                    new_upper_tri[i][j].append( np.dot(np.transpose(old_upper_tri[i][k]), old_upper_tri[j][k]) )
                    
            if new_upper_tri[j].has_key(i):
                if old_upper_tri[i][0]:
                    new_upper_tri[i][j].append( old_diag[i][1] * old_upper_tri[j][i] )
                else:
                    new_upper_tri[i][j].append( np.dot(np.transpose(old_diag[i][1]), old_upper_tri[j][i]) )
                    
            if len(new_upper_tri[i][j]) > 0:        
                new_upper_tri[i][j] = shape_preserving_sim(new_upper_tri[i][j])
            else:
                new_upper_tri[i].pop(j)

    # Diagonal
    for s in stochastics:
        new_diag[s] = (False,[])

        for ot in old_upper_tri[s].itervalues():
            new_diag[s][1].append(np.dot(np.transpose(ot), ot))
            
        if len(new_diag[s][1]) > 0:            
            new_diag[s][0] = False            
            if old_diag[s][0]:
                new_diag[s][1].append(np.diag(old_diag[s][1]**2))            
            else:
                new_diag[s][1].append(np.dot(old_diag[s][1], old_diag[s][1]))                
            new_diag[s][1] = shape_preserving_sim(new_diag[s][1])
            
        else:
            if old_diag[s][0]:
                new_diag[s][0] = True
                new_diag[s][1] = old_diag[s]**2
            else:
                new_diag[s][0] = False
                new_diag[s][1] = np.dot(old_diag[s], old_diag[s])

    return stochastic_keyed_dict_mat(new_diag, new_upper_tri, stochastics)
