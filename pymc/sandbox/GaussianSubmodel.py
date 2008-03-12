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
    
def assign_from_sparse(spvec, slices):
    for slice in slices.iteritems():
        slice[0].value = spvec[slice[1]]

def slice_by_stochastics(spmat, stochastics_i, stochastics_j, slices, stochastic_len):

    mat_list = []
    Ni = len(stochastics_i)
    Nj = len(stochastics_j)

    symm = stochastics_i is stochastics_j

    for i in xrange(Ni):
        mat_list.append([])
        si = stochastics_i[i]
        li = stochastic_len[si]

        if symm:
            # Superdiagonal
            for j in xrange(i):
                sj = stochastics_i[j]
                mat_list[i].append(spmat[slices[sj], slices[si]])

            # Diagonal
            mat_list[i].append(spmat[slices[si], slices[si]])

            # Subdiagonal
            for j in xrange(i+1,Ni):
                sj = stochastics_i[j]
                lj = stochastic_len[sj]
                mat_list[i].append(cvx.base.spmatrix([],[],[], (lj,li)))

        else:
            for j in xrange(Nj):
                
                if slices[si].start < slices[stochastics_j[j]].start:
                    mat_list[i].append(spmat[slices[si], slices[stochastics_j[j]]].trans())
                
                else:
                    mat_list[i].append(spmat[slices[stochastics_j[j]], slices[si]])
                    
    return cvx.base.sparse(mat_list)    

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
                
    
    def compute_diag_chol_facs(self):
        """
        Computes the square root or Cholesky decomposition
        of the precision of each stochastic conditional on its parents.
        
        TODO: Use Deterministics, to avoid computing Cholesky factors
        unnecessarily.
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
    
    def get_A(self, stochastic):
        A = {}
        for c in stochastic.children:

            if c.__class__ is LinearCombination:
                for cc in c.extended_children:
                    A[cc] = 0.
                    for elem in c.coefs[stochastic]:
                        if c.sides[stochastic] == 'L':
                            A[cc] -= elem.value.T
                        else:
                            A[cc] -= elem.value


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
        
        TODO: Assemble tau_chol in coordinate form. All the millions of 
        empty sparse matrices really bog things down for large N.
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
        
        # Assemble and square sparse precision matrix.
        self.tau_chol = cvx.base.sparse(mat_list)
        self.tau = cvx.base.spmatrix([],[],[], (self.len,self.len))
        
        cvx.cholmod.options['supernodal'] = 1
        
        cvx.base.syrk(self.tau_chol, self.tau, uplo='U', trans='T')            

    def compute_changeable_mean(self):
        """
        Computes joint 'canonical mean' parameter:
        joint precision matrix times joint mean.
        """
        
        # Assemble mean vector
        mean = cvx.base.matrix(0.,size=(self.len, 1))
        
        for i in xrange(len(self.stochastic_list)-1,-1,-1):

            s = self.stochastic_list[i]

            mu_now = s.parents['mu']
            
            # If parent is a Stochastic
            if isinstance(mu_now, Stochastic):
                if mu_now.__class__ in gaussian_classes:
                    # If it's Gaussian, record its mean
                    mean[self.slices[s]] = mean[self.slices[mu_now]]
                else:
                    # Otherwise record its value.
                    mean[self.slices[s]] = s.parents.value['mu']

            # If parent is a LinearCombination
            elif isinstance(mu_now, LinearCombination):
                
                for j in xrange(len(mu_now.x)):

                    # For those elements that are Gaussian,
                    # add in the corresponding coefficient times
                    # the element's mean

                    if mu_now.x[j].__class__ in gaussian_classes:
                        mean[self.slices[s]] += np.dot(mean[self.slices[mu_now.x[j]]], mu_now.y[j])

                    elif mu_now.y[j].__class__ in gaussian_classes:
                        mean[self.slices[s]] += np.dot(mu_now.x[j], mean[self.slices[mu_now.y[j]]])
                        
                    else:
                        mean[self.slices[s]] += np.dot(mu_now.x[j], mu_now.y[j])
            else:
                mean[self.slices[s]] = s.parents.value['mu']
        
        self.mean = mean   
        
        # Multiply mean by precision
        full_eta = cvx.base.matrix(0.,size=(self.len, 1))
        cvx.base.symv(self.tau, mean, full_eta, uplo='U', alpha=1., beta=0.)

        # Slice canonical eta parameter by changeable stochastics.
        eta = cvx.base.matrix(0.,size=(self.changeable_len, 1))
        for s in self.changeable_stochastic_list:
            eta[self.changeable_slices[s]] = full_eta[self.slices[s]]            
        
        # Values of 'data'
        x = cvx.base.matrix(0.,size=(self.fixed_len, 1))
        for s in self.fixed_stochastic_list:
            x[self.fixed_slices[s]] = s.value
        
        # Slice tau.
        tau_offdiag = slice_by_stochastics(self.tau, self.changeable_stochastic_list, 
            self.fixed_stochastic_list, self.slices, self.stochastic_len)

        # print "WARNING: You're multiplying eta by 0 here because there's a screw-up somewhere upstream."
        # eta *= 0.
        # x *= -1.

        # Condition canonical eta parameter.
        cvx.base.gemv(tau_offdiag, x, eta, alpha=-1., beta=1., trans='T')

        self.x=x
        self.full_eta = full_eta
        self.tau_offdiag = tau_offdiag

        self.changeable_mean = np.asarray(self.backsolver(eta, squared=True)).squeeze()

    def compute_changeable_tau_chol(self):
        """
        Compute Cholesky decomposition of self's joint precision,
        sliced for stochastics.
        
        Slicing precision matrices conditions, it doesn't marginalize.
        """
        cvx.cholmod.options['supernodal'] = 1
        
        mat_list = []
        
        self.changeable_tau_slice = slice_by_stochastics(self.tau, self.changeable_stochastic_list, self.changeable_stochastic_list, self.slices, self.stochastic_len)

        self.backsolver = spmat_to_backsolver(self.changeable_tau_slice, self.changeable_len)
        
    def draw_conditional(self):
        """
        Sets values of stochastics in tau_slice_chol's keys to new
        values drawn conditional on rest of model.
        """ 
        dev = cvx.base.matrix(np.random.normal(size=self.changeable_len))
        dev = np.asarray(self.backsolver(dev)).squeeze()
        dev += self.changeable_mean
        assign_from_sparse(dev, self.changeable_slices)
        
        
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
                            for i in xrange(len(c.x)):
                                
                                if c.x[i].__class__ in gaussian_classes and c.y[i].__class__ in gaussian_classes:
                                    raise ValueError, 'Stochastics %s and %s are multiplied in LinearCombination %s. \
                                                        They cannot be in the same Gassian submodel.' % (c.x[i], c.y[i], c)

                                if sum([x is s for x in c.x]) + sum([y is s for y in c.y]) > 1:
                                    raise ValueError, 'Stochastic %s cannot appear more than once in the terms of \
                                                        LinearCombination %s.' % (s,c)
    
                        else:
                            raise ValueError, 'Stochastic %s has a parent %s which is Deterministic, but not\
                                                LinearCombination, which has extended children.' % (s,c)
                
    
        if not all([d.__class__ is LinearCombination for d in self.deterministics]):
            raise ValueError, 'All deterministics must be LinearCombinations.'

                                
if __name__=='__main__':
    
    
    
    from pylab import *
    
    import numpy as np
    
    # # =========================================
    # # = Test case 1: Some old smallish model. =
    # # =========================================
    # A = Normal('A',1,1)
    # B = Normal('B',A,2*np.ones(2))
    # C_tau = np.diag([.5,.5])
    # C_tau[0,1] = C_tau[1,0] = .25
    # C = MvNormal('C',B, C_tau,isdata=True)
    # D_mean = LinearCombination('D_mean', x=[np.ones((3,2))], y=[C])
    # 
    # D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    # # D = Normal('D',D_mean,.5*np.ones(3))
    # G = GaussianSubmodel([B,C,A,D,D_mean])
    # # G = GaussianSubmodel([A,B,C])
    # G.compute_diag_chol_facs()
    # G.compute_tau_chol()
    # 
    # G.compute_changeable_tau_chol()
    # G.compute_changeable_mean()
    # G.draw_conditional()
    # 
    # dense_tau = sp_to_ar(G.changeable_tau_slice)
    # for i in xrange(dense_tau.shape[0]):
    #     for j in xrange(i):
    #         dense_tau[i,j] = dense_tau[j,i]
    # CC=(dense_tau).I
    # sig_tau = np.linalg.cholesky(dense_tau)
    # 
    
    # ================================
    # = Test case 2: Autoregression. =
    # ================================
    N = 100
    W = np.eye(2)*N
    # W[0,1] = W[1,0] = .5
    x_list = [MvNormal('x_0',np.ones(2)*3,W,value=np.zeros(2))]
    for i in xrange(1,N):
        # L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
        x_list.append(MvNormal('x_%i'%i,x_list[i-1],W))
    
    # W = N
    # x_list = [Normal('x_0',1.,W,value=0)]
    # for i in xrange(1,N):
    #     # L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
    #     x_list.append(Normal('x_%i'%i,x_list[i-1],W))
    
    
    x_list[-1].value = x_list[-1].value * 0. + 1.
    x_list[N/2].isdata=True
    
    G = GaussianSubmodel(x_list)
    C = Container(x_list)
    G.compute_diag_chol_facs()
    G.compute_tau_chol()
    G.compute_changeable_tau_chol()
    G.compute_changeable_mean()
    
    dense_tau = sp_to_ar(G.tau)
    for i in xrange(dense_tau.shape[0]):
        for j in xrange(i):
            dense_tau[i,j] = dense_tau[j,i]
    CC=(dense_tau).I
    sig_tau = np.linalg.cholesky(dense_tau)
    
    clf()
    for i in xrange(10):
        G.draw_conditional()
        # G.draw_prior()
        
        # for x in x_list:
        #     x.random()
    
        plot(array(C.value))
        # plot(hstack(C.value))
    
        # dev = np.random.normal(size=2.*N)
        # plot(np.linalg.solve(sig_tau.T, dev)[::-2])
        
    