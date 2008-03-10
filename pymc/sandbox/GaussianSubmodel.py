# Need to assemble full precision matrix, preferably keep it in LIL for 
# flexible slicing. Bummer. Then take slices, convert to CSR, pass to
# sparse Cholesky decomposition function.
# 
#
# cvxopt.cholmod.symbolic, numeric to get factor
# cvxopt.cholmod.solve, spsolve to backsolve (very flexible)
# cvxopt.base.syrk to square the matrix.
# 
# Just need to convert to and from Cholmod matrix.
# Can make sparse matrices from blocks using cvxopt.

# TODO: non-Gaussian parents need to be added to offset
# TODO: Make get_offset method.
# TODO: Fill in actual functionality.

from pymc import *
import numpy as np
from graphical_utils import *
import cvxopt as cvx
from cvxopt import base, cholmod
# from dict_linalg_utils import *

gaussian_classes = [Normal, MvNormal, MvNormalCov, MvNormalChol]

def sp_to_ar(sp):
    """
    Debugging utility function that converts cvxopt sparse matrices
    to numpy matrices.
    """
    shape = sp.size
    ar = np.asmatrix(np.empty(shape))
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            ar[i,j] = sp[i,j]
    return ar

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
        self.stochastic_list = order_stochastic_list(self.stochastics | self.data_stochastics)
        self.N_stochastics = len(self.stochastic_list)

        # Need to figure out children and parents of model.
        self.children, self.parents = find_children_and_parents(self.stochastic_list)
        
        self.stochastic_indices, self.stochastic_len, self.slices, self.len\
         = ravel_submodel(self.stochastic_list)
        
        self.internal_stochastic_list = []
        for stochastic in self.stochastic_list:
            if not stochastic in self.children:
                self.internal_stochastic_list.append(stochastic)
            
        self.internal_stochastic_indices, self.internal_stochastic_len, self.internal_slices, self.internal_len\
        = ravel_submodel(self.internal_stochastic_list)
                
    
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
                diag = True
                chol_now = np.empty(np.atleast_1d(s.value).shape)
                chol_now.fill(np.sqrt(parent_vals['tau']))
                
            else:
                diag = False    
                if isinstance(s, MvNormal):
                    chol_now = np.linalg.cholesky(parent_vals['tau'])
    
                # Make the next two less stupid!
                # There are lapack routines for inverse-from-cholesky...
                # is that the best you can do? Probably not.
                elif isinstance(s, MvNormalCov):
                    chol_now = np.linalg.cholesky(np.linalg.inv(parent_vals['C']))
    
                elif isinstance(s, MvNormalChol):
                    chol_now = np.linalg.cholesky(np.linalg.inv(np.dot(parent_vals['sig'], parent_vals['sig'].T)))
    
            self.diag_chol_facs[s] = (diag, np.atleast_1d(chol_now).T)
            
            
    # def compute_canon_mean(self, tau_chol):
    #     """
    #     Computes joint precision * joint mean.
    #     """
    #     b_vec = empty(self.len)
    # 
    #     for s in self.stochastic_list:
    #         chol_now = self.diag_chol_facs[s]
    #         
    #         if isinstance(s, Normal):
    #             b_vec[self.slices[s]] /= chol_now
    #             
    #         else:
    #             flib.dtrsm_wrap(chol_now, b_vec[self.slices[s]], 'L', 'T', 'L')
    #         
    #     self.canon_mean = trisolve(tau_chol, b_vec)
    #         
    #         
    
    def get_A(self, stochastic):
        A = {}
        for c in stochastic.children:

            if c.__class__ is LinearCombination:
                for cc in c.extended_children:
                    A[cc] = -c.coefs[stochastic]
            else:
                if stochastic is c.parents['mu']:
                    if self.stochastic_len[c] == self.stochastic_len[stochastic]:
                        A[c] = -np.eye(self.stochastic_len[stochastic])
                    else:
                        A[c] = -np.ones(self.stochastic_len[c])
        return A
    
    def compute_tau_chol(self):
        """
        Computes Cholesky factor of joint precision matrix, 
        and stores it as a cvxopt sparse matrix.
        """
        # dtrsm_wrap(a,b,side,transa,uplo)
        # dtrmm_wrap(a,b,side,transa,uplo)

        mat_list = []
        for i in xrange(self.N_stochastics):
            mat_list.append([])
            
        for i in xrange(self.N_stochastics):
            
            si = self.stochastic_list[i]
            li = self.stochastic_len[si]
            
            A = self.get_A(si)
            
            # Append off-diagonals            
            for j in xrange(i):
                
                sj = self.stochastic_list[j]
                lj = self.stochastic_len[sj]
                # mat_list[i].append(cvx.base.spmatrix([],[],[], (lj, li)))

                
                # If j is a parent of s,
                if A.has_key(sj):

                    chol_j = self.diag_chol_facs[sj]

                    # If this parent's precision matrix is diagonal
                    if chol_j[0]:
                        A[sj] = (chol_j[1] * A[sj].T).T
                    
                    # If this parent's precision matrix is not diagonal
                    else:
                        flib.dtrmm_wrap(chol_j[1], A[sj], side='L', transa='N', uplo='U')
                    
                    mat_list[i].append(cvx.base.matrix(A[sj]))
                    
                else:
                    mat_list[i].append(cvx.base.spmatrix([],[],[], (lj, li)))
                    
            chol_i = self.diag_chol_facs[si]
            # Append diagonal
            if chol_i[0]:
                mat_list[i].append(cvx.base.spmatrix(chol_i[1], range(len(chol_i[1])), range(len(chol_i[1]))))
            else:
                mat_list[i].append(cvx.base.matrix(chol_i[1]))
                
            # Append zeros to end
            for j in xrange(i+1,self.N_stochastics):
                sj = self.stochastic_list[j]
                lj = self.stochastic_len[sj]
                mat_list[i].append(cvx.base.spmatrix([],[],[], (lj, li)))
        
        self.tau_chol = cvx.base.sparse(mat_list)
        self.tau = cvx.base.spmatrix([],[],[], (self.len,self.len))
        cvx.base.syrk(self.tau_chol, self.tau, uplo='U', trans='T')

        # self.tau = self.tau_chol.T * self.tau_chol

        # return mat_list
    
    def slice_by_stochastics(spmat, stochastics, slices, keep_stochastics):
        # Need to look into cvxopt fancy slicing.
        return stochastics, slices, self.len
        
    def assign_from_sparse(spvec, slices):
        for slice in slices.iterkeys():
            slice[0].value = spvec[slice[1]]
            

    def tau_slice_chol(self, stochastics):
        """
        Compute Cholesky decomposition of self's joint precision,
        sliced for stochastics.
        
        Slicing precision matrices conditions, it doesn't marginalize.
        """
        pass
    
    def draw_conditional(self, tau_slice_chol):
        """
        Sets values of stochastics in tau_slice_chol's keys to new
        values drawn conditional on rest of model.
        
        This is essentially a manual triangular backsolve, written in 
        Python. Uses dtrsm.
        """ 
        pass   
                
    # 
    # 
    # def _conditional_draw(self, stochastics, partial_slices, tot_len):
    #     """
    #     Draws random values for some stochastics conditional on parents
    #     and children.
    #     """
    # 
    # 
    # def _unconditional_draw(self, stochastics, partial_slices, tot_len):
    #     """
    #     Draws random values for some stochastics conditional on
    #     parents.
    #     """
    # 
    #     fullvec = np.random.normal(size=self.len)
    #     fullvec = self.backsolver(fullvec,trans='T')
    #     
    #     out_vec = np.empty(tot_len)
    #     
    #     for stochastic in stochastics:
    #         out_vec[partial_slices[stochastic]] = fullvec[self.slices[stochastic]]                
    # 
    #     set_ravelled_stochastic_values(out_vec, stochastics, partial_slices)
    # 
    # 
    # def _marginal_covariance(self, stochastics, tot_len):
    #     """
    #     Returns the marginal covariance of stochastics.
    #     
    #     NB for some reason this is currently returning the covariance 
    #     of the group given their parents... also useful I guess, but not
    #     what it's supposed to do.
    #     """
    #     cov_out = sp.lil_matrix((tot_len,tot_len))
    #     back = np.zeros((self.len, tot_len))
    #     out = np.empty((tot_len,tot_len))
    #     
    #     start_sofar = 0
    #     for stochastic in stochastics:
    #         back[self.slices[stochastic], start_sofar:start_sofar + self.stochastic_len[stochastic]] \
    #             = eye(self.stochastic_len[stochastic])
    #         start_sofar += self.stochastic_len[stochastic]
    # 
    #             
    #     back = self.backsolver(back, trans='T')
    #     back = dot(back,back.T)
    #     
    #     start_sofar = 0
    #     
    #     for stochastic in stochastics:
    #         this_len = self.stochastic_len[stochastic]
    #         
    #         other_start_sofar = 0
    #         for other_stochastic in stochastics:
    #             other_len = self.stochastic_len[other_stochastic]
    #             out[start_sofar : start_sofar + this_len, other_start_sofar:other_start_sofar + other_len] =\
    #             back[self.slices[stochastic], self.slices[other_stochastic]]
    #             other_start_sofar += other_len
    #             
    #         start_sofar += this_len
    #         
    #     return out
    # 
    # def _internal_tau_chol(self):
    #     """
    #     Wrapper for _partial_tau_chol where stochastics is self's
    #     internal stochastics.
    #     
    #     This should be wrapped in a Deterministic eventually.
    #     """
    #     self.internal_tau_chol = self._partial_tau_chol(self.internal_stochastic_list, self.internal_slices, self.internal_len)
    #     
    #     
    # def draw_from_prior(self):
    #     """
    #     Draw values for all internal stochastics conditional on children.
    #     """
    #     self._unconditional_draw(self.internal_stochastic_list, self.internal_slices, self.internal_len)
    #     
    #     
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
    
    from pylab import *
    
    A = Normal('A',0,1)
    B = Normal('B',A,2*np.ones(2))
    C_tau = np.diag([.5,.5])
    C_tau[0,1] = C_tau[1,0] = .25
    C = MvNormal('C',B, C_tau, isdata=True)
    D_mean = LinearCombination('D_mean', x=[C], coefs={C:np.ones((3,2))}, offset=0)
    
    D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    # D = Normal('D',D_mean,.5*np.ones(3))
    G = GaussianSubmodel([B,C,A,D,D_mean])
    # G = GaussianSubmodel([A,B,C])
    G.compute_diag_chol_facs()
    G.compute_tau_chol()
    dense_tau = sp_to_ar(G.tau)
    for i in xrange(G.len):
        for j in xrange(i):
            dense_tau[i,j] = dense_tau[j,i]
    CC=(dense_tau).I
    # print p
    # print 
    print CC
    
    # 
    # G.compute_diag_chol_facs()
    # G.compute_tau_chol()
    # q = G.tau_chol
    # p=q.todense()
    # print (p*p.T).I
    
    # W = np.eye(2)
    # # W[0,1] = W[1,0] = .5
    # x_list = [MvNormal('x_0',np.zeros(2),W)]
    # for i in xrange(1,100):
    #     # L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
    #     x_list.append(MvNormal('x_%i'%i,x_list[i-1],W))
    # # q = lam_dtrm('q',lambda x=x_list[999]: x**2)
    # # h = Gamma('h', alpha=q, beta=1)
    # G = GaussianSubmodel(x_list)
    # C = Container(x_list)
    # clf()
    # for i in xrange(10):
    #     G.draw_from_prior()
    #     # for x in x_list[:-1]:
    #     #     x.random()
    #     plot(array(C.value))