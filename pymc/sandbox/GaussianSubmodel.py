# TODO: non-Gaussian parents need to be added to offset
# TODO: Make get_offset method.
# TODO: Fill in actual functionality.

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
            stochastic_len[stochastic] = len(np.ravel(stochastic.value))
        else:
            stochastic_len[stochastic] = 1
        slices[stochastic] = slice(_len, _len + stochastic_len[stochastic])
        _len += stochastic_len[stochastic]

        # Record indices that correspond to each stochastic.
        for j in xrange(len(np.ravel(stochastic.value))):
            stochastic_indices.append((stochastic, j))

    return stochastic_indices, stochastic_len, slices, _len

def find_children_and_parents(stochastic_list):
    children = []
    parents = []
    for s in stochastic_list:
        if all([not child in stochastic_list for child in s.extended_children]):
            children.append(s)
        if all([not parent in stochastic_list for parent in s.extended_parents]):
            parents.append(s)
        
    return set(children), set(parents)
    
        

class GaussianSubmodel(ListTupleContainer):
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
        ListTupleContainer.__init__(self, input)
        self.check_input()

        self.generations = find_generations(self, with_data = True)
        self.stochastic_list = []
        for generation in self.generations[::-1]:
            self.stochastic_list += list(generation)

        # Need to figure out children and parents of model.
        self.children, self.parents = find_children_and_parents(self.stochastic_list)
        
        self.stochastic_indices, self.stochastic_len, self.slices, self.len\
         = ravel_submodel(self.stochastic_list)
        
    def get_A(self, stochastic):
        """
        Need to use LinearCombination here eventually.
        """
        A = []
        p = stochastic.parents['mu']
        if p.__class__ in gaussian_classes:
            A.append((self.slices[p], -np.eye(self.stochastic_len[p])))
        elif p.__class__ is LinearCombination:
            for pp in p.x:
                A.append((self.slices[pp], -p.coefs[pp]))
            
        return A
        

    def compute_diag_chol_facs(self):
        """
        Computes the square root or Cholesky decomposition
        of the precision of each stochastic conditional on its parents.
        """
        self.diag_chol_facs = {}
        
        for s in self.stochastic_list:

            parent_vals = s.parents.value

            if isinstance(s, Normal):
                # This is likely to be a bottleneck. Leaving it in for now for the sake of
                # getting stuff working.
                chol_now = np.empty(np.atleast_1d(s.value).shape)
                chol_now.fill(np.sqrt(parent_vals['tau']))
                
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
                if len(A) > 0:
                    for A_elem in A:
                        tau_chol[A_elem[0], self.slices[s]] = np.transpose(A_elem[1]) * chol_now
            
                start_i = self.slices[s].start
                for i in xrange(start_i, self.slices[s].stop):
                    tau_chol[i,i] = chol_now[i-start_i]
                    
            else:    

                tau_chol[self.slices[s], self.slices[s]] = chol_now

                if len(A) > 0:
                    for A_elem in A:
                        flib.dtrmm_wrap(chol_now, A_elem[1], side='L', transa='T', uplo='L')
                        tau_chol[A_elem[0], self.slices[s]] = np.transpose(A_elem[1])

        return sparse.csr_matrix(tau_chol)

        
    def check_input(self):
        """
        Improve this...
        """

        if not all([s.__class__ in gaussian_classes for s in self.stochastics]):
            raise ValueError, 'All stochastics must be Normal, MvNormal, MvNormalCov or MvNormalChol.'
        
        for s in self.stochastics:
            
            # Make sure all extended children are Gaussian.
            for c in s.extended_children:
                if c.__class__ in gaussian_classes:
                    if c in s.children:
                        if not s is c.parents['mu']:
                            raise ValueError, 'Stochastic %s is a non-mu parent of stochastic %s' % (s,c)
                else:
                    raise ValueError, 'Stochastic %s has non-Gaussian extended child %s' % (s,c)
            
            # Make sure all children that aren't Gaussian but have extended children are LinearCombinations.
            for c in s.children:
                if isinstance(c, Deterministic):
                    if len(c.extended_children) > 0:
                        if c.__class__ is LinearCombination:
                            if any([val is s for val in c.coefs.values()]) or s is c.offset:
                                raise ValueError, 'Stochastic %s is considered either a coefficient or an offset by\
                                                    LinearCombination %s. Must be considered an "x" value.' % (s, c)

                        else:
                            raise ValueError, 'Stochastic %s has a parent %s which is Deterministic, but not\
                                                LinearCombination, which has extended children.' % (s,c)
                
                    
                
            
        if not all([d.__class__ is LinearCombination for d in self.deterministics]):
            raise ValueError, 'All deterministics must be LinearCombinations.'

        
                                
if __name__=='__main__':
    
    A = Normal('A',0,1)
    B = Normal('B',A,2*np.ones(2))
    C_tau = np.diag([.5,.5])
    C_tau[0,1] = C_tau[1,0] = .25
    C = MvNormal('C',B, C_tau, isdata=True)
    D_mean = LinearCombination('D_mean', x=[C], coefs={C:np.ones((3,2))}, offset=0)
    
    # D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    D = Normal('D',D_mean,.5*np.ones(3))
    G = GaussianSubmodel([B,C,A,D,D_mean])
    
    G.compute_diag_chol_facs()
    q=G.compute_tau_chol()
    p=q.todense()
    print (p*p.T).I
    
    # W = np.eye(2)
    # W[0,1] = W[1,0] = .5
    # x_list = [MvNormal('x_0',np.zeros(2),W)]
    # for i in xrange(1,1000):
    #     L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
    #     x_list.append(MvNormal('x_%i'%i,L,W))
    # # q = lam_dtrm('q',lambda x=x_list[999]: x**2)
    # # h = Gamma('h', alpha=q, beta=1)
    # G = GaussianSubmodel(x_list)
    # G.compute_diag_chol_facs()
    # q=G.compute_tau_chol()
    # 