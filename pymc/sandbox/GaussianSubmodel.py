from pymc import *
import numpy as np
from scipy import sparse

gaussian_classes = [Normal, MvNormal, MvNormalCov, MvNormalChol]

def ravel_submodel(stochastic_list):
    """
    Takes a list of stochastics and returns:
        - Indices corresponding to each, 
        - Length of each,
        - Slices corresponding to each,
        - Total length,

    """

    N_stochastics = len(stochastic_list)
    stochastic_indices = []
    stochastic_len = {}
    slices = {}

    _len = 0
    for i in xrange(len(stochastic_list)):

        stochastic = stochastic_list[i]

        # Inspect shapes of all stochastics and create stochastic slices.
        if isinstance(stochastic.value, np.ndarray):
            stochastic_len[stochastic] = np.len(np.ravel(stochastic.value))
        else:
            stochastic_len[stochastic] = 1
        slices[stochastic] = slice(_len, _len + stochastic_len[stochastic])
        _len += stochastic_len[stochastic]

        # Record indices that correspond to each stochastic.
        for j in xrange(len(np.ravel(stochastic.value))):
            stochastic_indices.append((stochastic, j))

    return stochastic_indices, stochastic_len, slices, _len


class GaussianSubmodel(SetContainer):
    """
    G = GaussianSubmodel(input)
    
    Input is a submodel consisting entirely of Normals, MvNormals, 
    MvNormalCovs, MvNormalChols and LinearCombinations. The normals 
    can only depend on each other in the mean: the mean of each must
    be a linear combination of others.
    
    Has the capacity to compute the joint canonical parameters of the
    submodel. The Cholesky factor of the joint precision matrix is
    stored as a sparse matrix for efficient conditionalization.
    
    Supports the following queries:

    - G.posterior(stochastics) : Cholesky factor of posterior precision
      and posterior mean of stochastics, conditional on parents and children
      of submodel.

    - G.full_conditional(stochastics) : Cholesky factor of precision and mean 
      of stochastics conditional on rest of submodel.
      
    - G.prior(stochastics) : Cholesky factor of precision and mean of stochastics
      conditional on parents of submodel
      
    - G.conditional(stochastics, evidence_stochastics) : Cholesky factor of 
      precision and mean of stochastics conditional on evidence_stochastics.
    """

    def __init__(self, input):
        SetContainer.__init__(self, input, name='GaussianSubmodel')
        self.check_input()

        self.generations = find_generations(self, with_data = True)
        self.stochastic_list = []
        for generation in self.generations[::-1]:
            self.stochastic_list += list(generation)
        # self.stochastic_list = self.stochastics | self.data_stochastics
        # Need to figure out children of model.

        self.stochastic_indices, self.stochastic_len, self.slices, self.len\
         = ravel_submodel(self.stochastic_list)

        
    def get_A(self, stochastic):
        """
        Need to use LinearCombination here eventually.
        """
        p = stochastic.parents['mu']
        if p in self.stochastic_list:
            return (self.slices[p], -np.eye(self.stochastic_len[p]))
        

    def compute_diag_chol_facs(self):
        """
        Computes the square root or Cholesky decomposition
        of the precision of each stochastic conditional on its parents.
        """
        self.diag_chol_facs = {}
        
        for s in self.stochastic_list:
            parent_vals = s.parents.value
            if isinstance(s, Normal):
                chol_now = np.sqrt(parent_vals['tau'])
                
            else:    
                if isinstance(s, MvNormal):
                    chol_now = np.linalg.cholesky(parent_vals['tau'])

                # Make the next two less stupid!
                # There are lapack routines for inverse-from-cholesky...
                # is that the best you can do? Probably not.
                elif isinstance(s, MvNormalCov):
                    chol_now = np.linalg.cholesky(np.linalg.inv(parent_vals['C']))

                elif isinstance(s, MvNormalChol):
                    chol_now = np.linalg.cholesky(np.linalg.inv(np.dot(parent_vals['sig'], parent_vals['sig'].T)))

            self.diag_chol_facs[s] = np.atleast_1d(chol_now)
            
            
    def compute_canon_mean(self, tau_chol):
        """
        Computes joint precision * joint mean.
        """
        b_vec = empty(self.len)

        for s in self.stochastic_list:
            chol_now = self.diag_chol_facs[s]
            
            if isinstance(s, Normal):
                b_vec[self.slices[s]] /= chol_now
                
            else:
                flib.dtrsm_wrap(chol_now, b_vec[self.slices[s]], 'L', 'T', 'L')
            
        return trisolve(tau_chol, b_vec)
            
            
    def compute_tau_chol(self):
        """
        Computes Cholesky factor of joint precision matrix in sparse (csc) storage.
        """
        # dtrsm_wrap(a,b,side,transa,uplo)
        # dtrmm_wrap(a,b,side,transa,uplo)

        
        # Eventually ditch lil_matrix.
        tau_chol = sparse.lil_matrix((self.len, self.len))
        
        for s in self.stochastic_list:

            chol_now = self.diag_chol_facs[s]

            A = self.get_A(s)

            if isinstance(s, Normal):
                if A is not None:
                    tau_chol[A[0], self.slices[s]] = A[1] * chol_now
            
                start_i = self.slices[s].start
                for i in xrange(start_i, self.slices[s].stop):
                    tau_chol[i,i] = chol_now[i-start_i]
                    
            else:    

                tau_chol[self.slices[s], self.slices[s]] = chol_now
                if A is not None:
                    flib.dtrmm_wrap(chol_now, A[1], side='R', transa='N', uplo='L')
                    tau_chol[self.slices[s], A[0]] = A[1]

        return sparse.csr_matrix(tau_chol)

        
    def check_input(self):
        """
        Improve this...
        """
        
        if not all([s.__class__ in gaussian_classes for s in self.stochastics]):
            raise ValueError, 'All stochastics must be Normal, MvNormal, MvNormalCov or MvNormalChol.'
            
        if not all([d.__class__ is LinearCombination for d in self.deterministics]):
            raise ValueError, 'All deterministics must be LinearCombinations.'

        
                                
if __name__=='__main__':
    
    A = Normal('A',0,1)
    B = Normal('B',A,2)
    C = Normal('C',B,.5, isdata=True)
    D = Normal('D',A,.5)
    G = GaussianSubmodel([D,B,C,A])

    G.compute_diag_chol_facs()
    q=G.compute_tau_chol()
    p=q.todense()
    print (p.T*p).I
    print (p*p.T).I
    
    h = np.asmatrix(np.eye(3))
    h[1,0] = -1
    h[2,1] = -1
    
    tau = np.asmatrix(np.diag([C.parents['tau'], B.parents['tau'], A.parents['tau']]))
    m = h*np.sqrt(tau)