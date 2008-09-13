# TODO: Specific submodel factories: DLM, LM.
# TODO: Optimization!
# TODO: Real test suite.
# TODO: Possibly base this on GDAGsim. Would make programming a little
# easier, but mostly would be a lighter dependency.

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from pymc import *
import copy as sys_copy
import numpy as np
from graphical_utils import *
import cvxopt as cvx
from cvxopt import base, cholmod

normal_classes = [Normal, MvNormal, MvNormalCov, MvNormalChol]

__all__ = ['normal_classes', 'sp_to_ar', 'assign_from_sparse', 'slice_by_stochastics', 'spmat_to_backsolver', 'crawl_normal_submodel', 'NormalSubmodel']

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
    
def assign_from_sparse(spvec, slices):
    for slice in slices.iteritems():
        slice[0].value = spvec[slice[1]]

# TODO: Pass in slices_from and slices_to, do for si in A.iterkeys() instead of for i in xrange(Ni).
# This should be much faster. It's tricky, though.
def slice_by_stochastics(spmat, stochastics_i, stochastics_j, slices_from, slices_to, stochastic_len, A):

    Ni = len(stochastics_i)
    Nj = len(stochastics_j)
    
    m = sum([stochastic_len[s] for s in stochastics_i])
    n = sum([stochastic_len[s] for s in stochastics_j])

    out = cvx.base.spmatrix([],[],[], (n,m))

    symm = stochastics_i is stochastics_j

    i_index = 0

    for i in xrange(Ni):

        j_index = 0
        si = stochastics_i[i]
        li = stochastic_len[si]
        i_slice = slice(i_index,i_index+li)
        i_index += li
        
        i_slice_to = slices_to[si]
        i_slice_from = slices_from[si]
        
        # print i_slice, i_slice_to

        if symm:
            Ai = A[si]
            # Superdiagonal
            
            for sj in Ai.iterkeys():
                if slices_to.has_key(sj):
                    out[slices_to[sj], i_slice_to] = spmat[slices_from[sj], i_slice_from]
            
            # for j in xrange(i):
            #     sj = stochastics_j[j]
            #     lj = stochastic_len[sj]
            #     j_slice = slice(j_index, j_index+lj)                
            #     j_index += lj
            #     
            #     if Ai.has_key(sj):
            #         out[j_slice, i_slice] = spmat[slices_from[sj], i_slice_from]

            # Diagonal
            out[i_slice_to,i_slice_to] = spmat[slices_from[si], slices_from[si]]


        else:
            for j in xrange(Nj):
                sj = stochastics_j[j]
                lj = stochastic_len[sj]
                j_slice = slice(j_index, j_index+lj)
                j_index += lj
                
                if slices_from[si].start < slices_from[stochastics_j[j]].start:
                    out[j_slice,i_slice] = spmat[slices_from[si], slices_from[sj]].trans()
                                               
                else:                          
                    out[j_slice,i_slice] = spmat[slices_from[sj], slices_from[si]]                

    print
    print 'slices_from:'
    for item in slices_from.iteritems():
        print '\t',item[0], item[1]
    print 'slices_to:'
    for item in slices_to.iteritems():
        print '\t',item[0], item[1]    
    print 'This should be all zeros when slices_from is the same as slices_to.'                
    print np.array(cvx.base.matrix(spmat - out))
    return out

def spmat_to_backsolver(spmat, N):
    # Assemple and factor sliced sparse precision matrix.
    chol = cvx.cholmod.symbolic(spmat, uplo='U')           
    cvx.cholmod.numeric(spmat, chol)

    # Find the diagonal part of the P.T L D L.T P factorization
    inv_sqrt_D = cvx.base.matrix(np.ones(N))
    cvx.cholmod.solve(chol, inv_sqrt_D, sys=6)
    inv_sqrt_D = cvx.base.sqrt(inv_sqrt_D)
    
    # Make function that backsolves either the Cholesky factor or its transpose
    # against an input vector or matrix.
    def backsolver(dev, uplo='U', squared=False, inv_sqrt_D = inv_sqrt_D, chol=chol):
        
        if uplo=='U':
            if squared:
                cvx.cholmod.solve(chol, dev)
            else:
                dev = cvx.base.mul(dev , inv_sqrt_D)
                cvx.cholmod.solve(chol, dev, sys=5)
                cvx.cholmod.solve(chol, dev, sys=8)

        elif uplo=='L':
            if squared:
                cvx.cholmod.solve(chol, dev)
            else:
                cvx.cholmod.solve(chol, dev, sys=7)
                cvx.cholmod.solve(chol, dev, sys=4)                
                dev = cvx.base.mul(dev , inv_sqrt_D)                

        return dev
        
    return backsolver

def crawl_normal_submodel(input):
    """
    Finds the biggest Normal submodel incorporating 'input.'
    
    TODO: Make this actually work...
    """
    input = Container(input)
    NormalSubmodel.check_input(input)
    output = input.stochastics
    
    return output
    
    
        
    
class NormalSubmodel(ListContainer):
    """
    G = NormalSubmodel(input)
    
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
        ListContainer.__init__(self, input)
        
        # Need to figure out children and parents of model.
        self.children, self.parents = find_children_and_parents(self.stochastics | self.data_stochastics)

        self.stochastic_list = order_stochastic_list(self.stochastics | self.data_stochastics)
        
        # TODO: Replace check_input with crawl(). If output of crawl is length 0,
        # THEN raise an error complaining that there are no eligible stochastics here.
        self.check_input(self.stochastic_list)                

        self.N_stochastics = len(self.stochastic_list)

        self.stochastic_indices, self.stochastic_len, self.slices, self.len\
         = ravel_submodel(self.stochastic_list)
        
        self.changeable_stochastic_list = []
        self.fixed_stochastic_list = []
        for stochastic in self.stochastic_list:
            if not stochastic in self.children and not stochastic.isdata:
                self.changeable_stochastic_list.append(stochastic)
            else:
                self.fixed_stochastic_list.append(stochastic)
            
        self.changeable_stochastic_indices, self.changeable_stochastic_len, self.changeable_slices, self.changeable_len\
        = ravel_submodel(self.changeable_stochastic_list)
        
        self.fixed_stochastic_indices, self.fixed_stochastic_len, self.fixed_slices, self.fixed_len\
        = ravel_submodel(self.fixed_stochastic_list)
        
        self.get_diag_chol_facs()
        self.get_A()
        self.get_mult_A()
        self.get_tau()       
        self.get_changeable_tau()
        # self.get_changeable_mean()
            
    def get_diag_chol_facs(self):
        """
        Creates self.diag_chol_facs, which is a list.
            Each element is a list of length 2.
                The first element is a boolean indicating whether this precision 
                  submatrix is diagonal.
                The second is a Deterministic whose value is the Cholesky factor 
                  (upper triangular) or square root of this precision submatrix.
        """
        self.diag_chol_facs = {}
        
        for s in self.stochastic_list:
    
            parent_vals = s.parents.value
            
            # Diagonal precision
            if isinstance(s, Normal):
                diag = True
                
                @deterministic
                def chol_now(tau=s.parents['tau'], d=s):
                    out = np.empty(np.atleast_1d(d).shape)
                    out.fill(np.sqrt(tau))
                    return out
            
            # Otherwise                    
            else:
                diag = False    
                if isinstance(s, MvNormal):
                    chol_now = Lambda('chol_now', lambda tau=s.parents['tau']: np.linalg.cholesky(tau).T)
    
                # Make the next two less stupid!
                elif isinstance(s, MvNormalCov):
                    chol_now = Lambda('chol_now', lambda C=s.parents['C']: np.linalg.cholesky(np.linalg.inv(C)).T)
    
                elif isinstance(s, MvNormalChol):
                    chol_now = Lambda('chol_now', lambda sig=s.parents['sig']: np.linalg.cholesky(np.linalg.inv(np.dot(sig, sig.T))).T)
    
            self.diag_chol_facs[s] = [diag, chol_now]
            
        self.diag_chol_facs = Container(self.diag_chol_facs)
    
    def get_A(self):
        """
        Creates self.A, which is a dictionary of dictionaries.
        
        A[si][sj], if present, is a Deterministic whose value is 
          -1 times the coefficient of si in the mean of sj.
        """

        self.A = {}
        for s in self.stochastic_list:
            self.A[s] = {}
            this_A = self.A[s]
            for c in s.children:

                if c.__class__ is LinearCombination:
                    for cc in c.children:

                        @deterministic
                        def A(coefs = c.coefs[s], side = c.sides[s]):
                            A = 0.
                        
                            for elem in coefs:
                                if side == 'L':
                                    A -= elem.T
                                else:
                                    A -= elem
                            return A
                            
                        this_A[cc] = A

                elif c.__class__ in normal_classes:
                    if s is c.parents['mu']:
                        if self.stochastic_len[c] == self.stochastic_len[s]:
                            this_A[c] = -np.eye(self.stochastic_len[s])
                        else:
                            this_A[c] = -np.ones(self.stochastic_len[c])
        
        self.A = Container(self.A)
    
    def get_mult_A(self):
        """
        Creates self.mult_A. This is just like self.A, but
        self.mult_A[si][sj] = self.diag_chol_facs[sj][1] * self.A[si][sj]
        """

        self.mult_A = {}    
        for i in xrange(self.N_stochastics):
            si = self.stochastic_list[i]
            this_A = self.A[si]
            self.mult_A[si] = {}
            this_mult_A = self.mult_A[si]
            
            for j in xrange(i):
        
                sj = self.stochastic_list[j]
        
                # If j is a parent of s,
                if this_A.has_key(sj):

                    chol_j = self.diag_chol_facs[sj]
                    
                    @deterministic
                    def mult_A(diag = chol_j[0], chol_j = chol_j[1], A = this_A[sj]):                        
                        # If this parent's precision matrix is diagonal
                        if diag:
                            out = (chol_j * A.T).T
            
                        # If this parent's precision matrix is not diagonal
                        else:
                            out = copy(A)
                            flib.dtrmm_wrap(chol_j, out, side='L', transa='N', uplo='U')

                        return out
                        
                    this_mult_A[sj] = mult_A
        
        self.mult_A = Container(self.mult_A)
    
    def get_tau(self):
        """
        Creates self.tau and self.tau_chol, which are Deterministics 
        valued as the joint precision matrix and its Cholesky factor,
        stored as cvxopt sparse matrices.
        """
        
        @deterministic
        def tau_chol(A = self.mult_A, diag_chol = self.diag_chol_facs):
            tau_chol = cvx.base.spmatrix([],[],[], (self.len,self.len))
        
            # TODO: for si in A.iterkeys(), etc. instead
            for si in A.iterkeys():
            
                this_A = A[si]
            
                # Append off-diagonals            
                for sj in this_A.iterkeys():
                                    
                    # If j is a parent of s,
                    tau_chol[self.slices[sj], self.slices[si]] = this_A[sj]
                    
                chol_i = diag_chol[si]
                chol_i_val = chol_i[1]

                # Write diagonal
                if chol_i[0]:
                    tau_chol[self.slices[si], self.slices[si]] = \
                      cvx.base.spmatrix(chol_i_val, range(len(chol_i_val)), range(len(chol_i_val)))
                else:
                    tau_chol[self.slices[si], self.slices[si]] = cvx.base.matrix(chol_i_val)
            return tau_chol
                
        
        # Square sparse precision matrix.
        @deterministic
        def tau(tau_chol = tau_chol):
            tau = cvx.base.spmatrix([],[],[], (self.len,self.len))
            cvx.base.syrk(tau_chol, tau, uplo='U', trans='T')    
            return tau
            
        self.tau, self.tau_chol = tau, tau_chol

    def get_changeable_mean(self):
        """
        Computes joint 'canonical mean' parameter:
        joint precision matrix times joint mean.
        """
        
        # Assemble mean vector
        mean_dict = {}
        
        for i in xrange(len(self.stochastic_list)-1,-1,-1):

            s = self.stochastic_list[i]

            mu_now = s.parents['mu']
            
            # If parent is a Stochastic
            if isinstance(mu_now, Stochastic):
                if mu_now.__class__ in normal_classes:
                    # If it's Normal, record its mean
                    
                    mean_dict[s] = mean_dict[mu_now]

                else:
                    # Otherwise record its value.
                    mean_dict[s] = mu_now
            
            # If parent is a LinearCombination
            elif isinstance(mu_now, LinearCombination):
                
                mean_terms = []
                for j in xrange(len(mu_now.x)):
            
                    # For those elements that are Normal,
                    # add in the corresponding coefficient times
                    # the element's mean
            
                    if mu_now.x[j].__class__ in normal_classes:
                        mean_var = Lambda('mean_var', lambda mu=mean_dict[mu_now.x[j]], s=mu_now.x[j]: np.resize(mu,np.shape(s)))
                            
                        mean_terms.append(Lambda('term', lambda x=mean_var, y=mu_now.coefs[mu_now.x[j]], s=s: 
                                                            np.reshape(np.dot(x,y), np.shape(s))))
            
                    elif mu_now.y[j].__class__ in normal_classes:
                        mean_var = Lambda('mean_var', lambda mu=mean_dict[mu_now.y[j]], s=mu_now.y[j]: np.resize(mu,np.shape(s)))

                        mean_terms.append(Lambda('term', lambda x=mu_now.coefs[mu_now.y[j]], y=mean_var, s=s: 
                                                            np.reshape(np.dot(x,y), np.shape(s))))
                        
                    else:
                        mean_terms.append(np.dot(mu_now.x[j], mu_now.y[j]))
                
                @deterministic
                def this_mean(mean_terms = mean_terms):
                    this_mean = 0.
                    for i in xrange(len(mean_terms)):
                        this_mean += mean_terms[i]
                    return this_mean
                mean_dict[s] = this_mean
                
            else:
                mean_dict[s] = mu_now
        
        self.mean_dict = Container(mean_dict)
        
        @deterministic
        def mean(mean_dict = self.mean_dict):
            mean = cvx.base.matrix(0.,size=(self.len, 1))
            for s in self.stochastic_list:
                mean[self.slices[s]] = mean_dict[s]
            return mean
        self.mean = mean
                
        # Multiply mean by precision
        @deterministic
        def full_eta(tau = self.tau, mean = mean):
            full_eta = cvx.base.matrix(0.,size=(self.len, 1))
            cvx.base.symv(tau, mean, full_eta, uplo='U', alpha=1., beta=0.)
            return full_eta
        self.full_eta = full_eta
        
        # Values of 'data'. This is a hack... fix it sometime.
        @deterministic
        def x(stochastics = self.fixed_stochastic_list):
            x = cvx.base.matrix(0.,size=(self.fixed_len, 1))
            for s in self.fixed_stochastic_list:
                x[self.fixed_slices[s]] = s.value
            return x
        self.x = x
        
        # Slice tau.
        @deterministic
        def tau_offdiag(tau = self.tau):
            return slice_by_stochastics(tau, self.changeable_stochastic_list, 
                self.fixed_stochastic_list, self.slices, self.changeable_slices, self.stochastic_len, self.A)
        self.tau_offdiag = tau_offdiag
        
        # Condition canonical eta parameter.
        # Slice canonical eta parameter by changeable stochastics.
        @deterministic(cache_depth=2)
        def eta(full_eta = full_eta, x=x, tau_offdiag = tau_offdiag):
            eta = cvx.base.matrix(0.,size=(self.changeable_len, 1))
            for s in self.changeable_stochastic_list:
                eta[self.changeable_slices[s]] = full_eta[self.slices[s]]            
            cvx.base.gemv(tau_offdiag, x, eta, alpha=-1., beta=1., trans='T')
            return eta

        self.eta = eta
            
        @deterministic
        def changeable_mean(backsolver = self.backsolver, eta = self.eta):
            return np.asarray(backsolver(sys_copy.copy(eta), squared=True)).squeeze()
        self.changeable_mean = changeable_mean

    def get_changeable_tau(self):
        """
        Creates self.changeable_tau_slice, which is a Deterministic valued
        as self.tau sliced according to self.changeable_stochastic_list,
        
        and self.backsolver, which solves linear equations involving
        self.changeable_tau_slice.
        """
        
        @deterministic
        def changeable_tau_slice(tau = self.tau):
            return slice_by_stochastics(tau, self.changeable_stochastic_list, 
                self.changeable_stochastic_list, self.slices, self.changeable_slices, self.stochastic_len, self.A)
        self.changeable_tau_slice = changeable_tau_slice

        @deterministic
        def backsolver(changeable_tau_slice = changeable_tau_slice):
            cvx.cholmod.options['supernodal'] = 1
            return spmat_to_backsolver(changeable_tau_slice, self.changeable_len)
        
        self.changeable_tau_slice, self.backsolver = changeable_tau_slice, backsolver
        
    def draw_conditional(self):
        """
        Sets values of stochastics in tau_slice_chol's keys to new
        values drawn conditional on rest of model.
        """ 
        dev = cvx.base.matrix(np.random.normal(size=self.changeable_len))
        dev = np.asarray(self.backsolver.value(dev)).squeeze()
        dev += self.changeable_mean.value
        assign_from_sparse(dev, self.changeable_slices)
        
    
    @staticmethod             
    def check_input(stochastics):
        """
        Improve this...
        """
    
        if not all([s.__class__ in normal_classes for s in stochastics]):
            raise ValueError, 'All stochastics must be Normal, MvNormal, MvNormalCov or MvNormalChol.'
        
        for s in stochastics:
            
            # Make sure all extended children are Normal.
            for c in s.extended_children:
                if c.__class__ in normal_classes:
                    if c in s.children:
                        if not s is c.parents['mu']:
                            raise ValueError, 'Stochastic %s is a non-mu parent of stochastic %s' % (s,c)
                else:
                    raise ValueError, 'Stochastic %s has non-Normal extended child %s' % (s,c)
            
            # Make sure all children that aren't Normal but have extended children are LinearCombinations.
            for c in s.children:
                if isinstance(c, Deterministic):
                    if len(c.extended_children) > 0:
                        if c.__class__ is LinearCombination:
                            for i in xrange(len(c.x)):
                                
                                if c.x[i].__class__ in normal_classes and c.y[i].__class__ in normal_classes:
                                    raise ValueError, 'Stochastics %s and %s are multiplied in LinearCombination %s. \
                                                        They cannot be in the same Gassian submodel.' % (c.x[i], c.y[i], c)

                                if sum([x is s for x in c.x]) + sum([y is s for y in c.y]) > 1:
                                    raise ValueError, 'Stochastic %s cannot appear more than once in the terms of \
                                                        LinearCombination %s.' % (s,c)
    
                        else:
                            raise ValueError, 'Stochastic %s has a parent %s which is Deterministic, but not\
                                                LinearCombination, which has extended children.' % (s,c)
                                
if __name__=='__main__':
    
    from pylab import *
    
    import numpy as np
    
    # =========================================
    # = Test case 1: Some old smallish model. =
    # =========================================
    A = Normal('A',1,1)
    B = Normal('B',A,2*np.ones(2))
    C_tau = np.diag([.5,.5])
    C_tau[0,1] = C_tau[1,0] = .25
    C = MvNormal('C',B, C_tau,isdata=True)
    D_mean = LinearCombination('D_mean', x=[np.ones((3,2))], y=[C])
    
    D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    # D = Normal('D',D_mean,.5*np.ones(3))
    G = NormalSubmodel([B,C,A,D,D_mean])
    # G = NormalSubmodel([A,B,C])
    G.draw_conditional()
    
    dense_tau = sp_to_ar(G.tau.value)
    for i in xrange(dense_tau.shape[0]):
        for j in xrange(i):
            dense_tau[i,j] = dense_tau[j,i]
    CC=(dense_tau).I
    sig_tau = np.linalg.cholesky(dense_tau)
    
    
    # # ================================
    # # = Test case 2: Autoregression. =
    # # ================================
    # 
    # 
    # N=100
    # W = Uninformative('W',np.eye(2)*N)
    # base_mu = Uninformative('base_mu', np.ones(2)*3)
    # # W[0,1] = W[1,0] = .5
    # x_list = [MvNormal('x_0',base_mu,W,value=np.zeros(2))]
    # for i in xrange(1,N):
    #     # L = LinearCombination('L', x=[x_list[i-1]], y = [np.eye(2)])
    #     x_list.append(MvNormal('x_%i'%i,x_list[i-1],W))
    # 
    # # W = N
    # # x_list = [Normal('x_0',1.,W,value=0)]
    # # for i in xrange(1,N):
    # #     # L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
    # #     x_list.append(Normal('x_%i'%i,x_list[i-1],W))
    # 
    # 
    # x_list[-1].value = x_list[-1].value * 0. + 1.
    # x_list[N/2].isdata=True
    # 
    # G = NormalSubmodel(x_list)
    # # C = Container(x_list)
    # #     
    # # dense_tau = sp_to_ar(G.tau.value)
    # # for i in xrange(dense_tau.shape[0]):
    # #     for j in xrange(i):
    # #         dense_tau[i,j] = dense_tau[j,i]
    # # CC=(dense_tau).I
    # # sig_tau = np.linalg.cholesky(dense_tau)
    # # 
    # # clf()
    # # for i in xrange(10):
    # #     G.draw_conditional()
    # #     # G.draw_prior()
    # #     
    # #     # for x in x_list:
    # #     #     x.random()
    # # 
    # #     plot(array(C.value))
    # #     # plot(hstack(C.value))
    # # 
    # #     # dev = np.random.normal(size=2.*N)
    # #     # plot(np.linalg.solve(sig_tau.T, dev)[::-2])
    # #     
    # # 