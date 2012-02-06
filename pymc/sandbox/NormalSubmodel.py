# Bottleneck is now slicing in two places: tau_chol, which fetches from 'A' and places into
# the Cholesky factor based on stochastics (not much to do about this); and tau_offdiag /
# changeable_tau, which doesn't yet slice efficiently with irregular strides. Hopefully
# the post to scipy gives some ideas.
# May want to construct sparse matrices directly in coordinate format in Pyrex in the future.
#
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
from IPython.Debugger import Pdb

from pymc import six
xrange = six.moves.xrange

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

def assign_from_sparse(spvec, stochastics, slices):
    for i in xrange(len(slices)):
        stochastics[i].value = spvec[slices[i]]

def greedy_slices(x, y):
    """
    x and y must be the same length. Will try to match long contiguous or strided blocks
    in x with long contiguous or strided blocks in y. Return value will be a tuple consisting
    of two lists of slices.
    """

    if len(x)==0:
        return ([slice(0,0)], [slice(0,0)])

    slices_x = []
    slices_y = []
    sx = x[0]
    sy = y[0]
    csx = x[1]-x[0]
    csy = y[1]-y[0]

    for i in xrange(1,len(x)):

        # Break current slice if x or y breaks stride
        if not (x[i]-x[i-1]==csx and y[i]-y[i-1]==csy):
            slices_x.append(slice(sx, x[i-1]+1, csx))
            slices_y.append(slice(sy, y[i-1]+1, csy))
            csx = x[i+1]-x[i]
            csy = y[i+1]-y[i]
            sx = x[i]
            sy = y[i]

    # Append final slice
    slices_x.append(slice(sx, x[-1]+1, csx))
    slices_y.append(slice(sy, y[-1]+1, csy))
    return (slices_x, slices_y)

def contiguize_slices(s_list, slices_from, slices_to):

    def none_to_1(x):
        if x is None:
            return 1
        else:
            return x

    # Find all indices needed
    ind_from = []
    ind_to = []
    for i in xrange(len(s_list)):
        sf = slices_from[i]
        st = slices_to[i]
        ind_from.extend(range(sf.start, sf.stop, none_to_1(sf.step)))
        ind_to.extend(range(st.start, st.stop, none_to_1(st.step)))
    ind_from = np.array(ind_from)
    ind_to = np.array(ind_to)

    # Greedily break indices into chunks
    return greedy_slices(ind_from, ind_to)

# def slice_by_stochastics(spmat, stochastics_i, stochastics_j, slices_from, slices_to, stochastic_len, mn):
#     """
#     Arguments:
#
#       - spmat : cvxopt sparse matrix
#         Matrix to be sliced
#       - stochastics_i : list
#         Stochastics determining row-indices of slice, in order.
#       - stochastics_j : list
#         Stochastics determining column-indices of slice, in order.
#       - slices_from : dictionary
#         spmat[slices_from[s1], slices_from[s2]] will give the slice of the precision
#         matrix corresponding to s1 and s2.
#       - slices_to : dictionary
#         Only used if not symm. Saves figuring out how to pack the output matrix
#         from stochastic_i and stochastic_j.
#       - stochastic_len : dictionary
#         stochastic_len[s] gives the length of s.
#       - mn : dictionary
#         mn[s] gives the moral neighbors of s. The only nonzero entries in the
#         precision matrix correspond to moral neighbors, so this can be used to
#         speed up the slicing.
#     """
#     Ni = len(stochastics_i)
#     Nj = len(stochastics_j)
#
#     m = sum([stochastic_len[s] for s in stochastics_i])
#     n = sum([stochastic_len[s] for s in stochastics_j])
#
#     out = cvx.base.spmatrix([],[],[], (n,m))
#
#     symm = stochastics_i is stochastics_j
#
#     i_index = 0
#     if not symm:
#         j_index = 0
#         j_slices = {}
#         for j in xrange(Nj):
#             sj = stochastics_j[j]
#             lj = stochastic_len[sj]
#             j_slices[sj] = slice(j_index, j_index+lj)
#             j_index += lj
#
#     for i in xrange(Ni):
#         si = stochastics_i[i]
#         li = stochastic_len[si]
#         i_slice = slice(i_index,i_index+li)
#         i_index += li
#         i_slice_to = slices_to[si]
#         i_slice_from = slices_from[si]
#
#         if symm:
#             # Superdiagonal
#             for sj in mn[si]:
#                 if slices_to.has_key(sj):
#                     out[slices_to[sj], i_slice_to] = spmat[slices_from[sj], i_slice_from]
#             # Diagonal
#             out[i_slice_to,i_slice_to] = spmat[i_slice_from, i_slice_from]
#         else:
#             for sj in mn[si]:
#                 if sj not in stochastics_j:
#                     continue
#                 j_slice_from = slices_from[sj]
#                 if i_slice_from.start < j_slice_from.start:
#                     out[j_slices[sj],i_slice] = spmat[i_slice_from, j_slice_from].trans()
#                 else:
#                     out[j_slices[sj],i_slice] = spmat[j_slice_from, i_slice_from]
#
#     return out

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
        self.children, self.parents = find_children_and_parents(self.stochastics | self.observed_stochastics)

        self.stochastic_list = order_stochastic_list(self.stochastics | self.observed_stochastics)

        self.check_input(self.stochastic_list)

        self.N_stochastics = len(self.stochastic_list)
        self.stochastic_list_numbers = {}
        for i in xrange(self.N_stochastics):
            self.stochastic_list_numbers[self.stochastic_list[i]] = i

        self.stochastic_indices, self.stochastic_len, self.slices, self.len\
         = ravel_submodel(self.stochastic_list)

        self.slice_dict = {}
        for i in xrange(self.N_stochastics):
            self.slice_dict[self.stochastic_list[i]] = self.slices[i]

        self.changeable_stochastic_list = []
        self.fixed_stochastic_list = []
        for stochastic in self.stochastic_list:
            if not stochastic in self.children and not stochastic.observed:
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
        self.get_mean_dict()
        self.get_changeable_mean()

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

        for i in xrange(self.N_stochastics):
            s = self.stochastic_list[i]
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

            self.diag_chol_facs[i] = [diag, chol_now]

        self.diag_chol_facs = Container(self.diag_chol_facs)

    def get_A(self):
        """
        Creates self.A, which is a dictionary of dictionaries.

        A[si][sj], if present, is a Deterministic whose value is
          -1 times the coefficient of si in the mean of sj.
        """

        self.A = np.zeros(self.N_stochastics, dtype=object)
        for i in xrange(self.N_stochastics):
            s = self.stochastic_list[i]
            self.A[i] = []
            this_A = self.A[i]
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

                        this_A.append((self.stochastic_list_numbers[cc],(A)))

                elif c.__class__ in normal_classes:
                    if s is c.parents['mu']:
                        if self.stochastic_len[c] == self.stochastic_len[s]:
                            this_A.append((self.stochastic_list_numbers[c],-np.eye(self.stochastic_len[s])))
                        else:
                            this_A.append((self.stochastic_list_numbers[c],-np.ones(self.stochastic_len[c])))

        self.A = Container(self.A)

    def get_mult_A(self):
        """
        Creates self.mult_A. This is just like self.A, but
        self.mult_A[si][sj] = self.diag_chol_facs[sj][1] * self.A[si][sj]
        """

        self.mult_A = np.zeros(self.N_stochastics, dtype=object)
        for i in xrange(self.N_stochastics):
            si = self.stochastic_list[i]
            this_A = self.A[i]
            self.mult_A[i] = []
            this_mult_A = self.mult_A[i]

            for j, A_elem in this_A:
                chol_j = self.diag_chol_facs[j]

                @deterministic
                def mult_A(diag = chol_j[0], chol_j = chol_j[1], A = A_elem):
                    # If this parent's precision matrix is diagonal
                    if diag:
                        out = (chol_j * A.T).T
                    # If this parent's precision matrix is not diagonal
                    else:
                        out = copy(A)
                        flib.dtrmm_wrap(chol_j, out, side='L', transa='N', uplo='U')

                    return out

                this_mult_A.append((j, mult_A))

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

            for i in xrange(self.N_stochastics):
                si = self.stochastic_list[i]
                i_slice = self.slices[i]
                this_A = A[i]

                chol_i = diag_chol[i]
                chol_i_val = chol_i[1]

                # Write diagonal
                if chol_i[0]:
                    tau_chol[i_slice, i_slice] = \
                      cvx.base.spmatrix(chol_i_val, range(len(chol_i_val)), range(len(chol_i_val)))
                else:
                    tau_chol[i_slice, i_slice] = cvx.base.matrix(chol_i_val)

                # Append off-diagonals
                for j, A_elem in this_A:
                    # If j is a parent of s,
                    tau_chol[self.slices[j], i_slice] = A_elem


            return tau_chol


        # Square sparse precision matrix.
        @deterministic
        def tau(tau_chol = tau_chol):
            tau = cvx.base.spmatrix([],[],[], (self.len,self.len))
            cvx.base.syrk(tau_chol, tau, uplo='U', trans='T')
            return tau

        self.tau, self.tau_chol = tau, tau_chol

    def get_mean_dict(self):
        """
        Forms self.mean_dict, which is a dictionary. self.mean_dict[x] for stochastic x is:
        - A constant or PyMC object if x is in the first generation of the normal submodel.
        - self.slices[x.parents['mu']] if x's direct parent is normal and in the normal
          submodel.
        - A list of (constant or PyMC object, slice) or
          (constant or PyMC object, constant or PyMC object) tuples if x's direct parent is a
          LinearCombination. In this case, in each element the slice corresponds to x's
          extended parent that is in the normal submodel (there should only be one per tuple),
          and the constant or PyMC object that multiplies that parent.
        """
        mean_dict = {}

        # for i in xrange(self.N_stochastics-1,-1,-1):
        for s in self.stochastic_list:

            mu_now = s.parents['mu']

            # If parent is in normal submodel
            if mu_now.__class__ in normal_classes:
                if mu_now in self.stochastic_list:
                    mean_dict[s] = ('n',self.slice_dict[mu_now])

            # If parent is a LinearCombination
            elif isinstance(mu_now, LinearCombination):

                mean_terms = []
                for j in xrange(len(mu_now.x)):

                    # For those elements that are Normal,
                    # add in the corresponding coefficient times
                    # the element's mean
                    if mu_now.x[j].__class__ in normal_classes:
                        if mu_now.x[j] in self.stochastic_list:
                            mean_terms.append(('l', self.slice_dict[mu_now.x[j]], mu_now.y[j]))

                    if mu_now.y[j].__class__ in normal_classes:
                        if mu_now.y[j] in self.stochastic_list:
                            mean_terms.append(('r', mu_now.x[j], self.slice_dict[mu_now.y[j]]))

                    else:
                        mean_terms.append(('n', mu_now.x[j], mu_now.y[j]))

                mean_dict[s] = ('l',mean_terms)

            else:
                mean_dict[s] = ('c',mu_now)

        self.mean_dict = mean_dict

    def get_changeable_mean(self):
        """
        Computes joint 'canonical mean' parameter:
        joint precision matrix times joint mean.
        """

        @deterministic
        def mean(mean_dict = self.mean_dict):
            mean = cvx.base.matrix(0.,size=(self.len, 1))

            for i in xrange(self.N_stochastics-1,-1,-1):
                s = self.stochastic_list[i]
                sl = self.slices[i]
                case, info = mean_dict[s]

                # Constant-parent case
                if case=='c':
                    mean[sl] = np.ravel(info)
                # Parent in normal submodel
                elif case=='n':
                    mean[sl] = mean[info]
                # Parent is LinearCombination
                else:
                    this_mean = np.zeros(sl.stop - sl.start)
                    for pair in info:
                        # Left-hand member is in normal submodel
                        if pair[0]=='l':
                            this_mean += np.ravel(np.dot(mean[pair[1]], pair[2]))
                        # Right-hand member is in normal submodel
                        if pair[0]=='r':
                            this_mean += np.ravel(np.dot(pair[1], mean[pair[2]]))
                        # Neither member is in normal submodel
                        else:
                            this_mean += np.ravel(np.dot(pair[1], pair[2]))
                    mean[sl] = this_mean
            return mean

        self.mean = mean

        # Multiply mean by precision
        @deterministic
        def full_eta(tau = self.tau, mean = mean):
            full_eta = cvx.base.matrix(0.,size=(self.len, 1))
            cvx.base.symv(tau, mean, full_eta, uplo='U', alpha=1., beta=0.)
            return full_eta
        self.full_eta = full_eta

        # FIXME: Values of 'data'. This is a hack... fix it sometime.
        @deterministic
        def x(stochastics = self.fixed_stochastic_list):
            x = cvx.base.matrix(0.,size=(self.fixed_len, 1))
            for i in xrange(len(self.fixed_stochastic_list)):
                x[self.fixed_slices[i]] = self.fixed_stochastic_list[i].value
            return x
        self.x = x


        # j_slice_from = slices_from[sj]
        # if i_slice_from.start < j_slice_from.start:
        #     out[j_slices[sj],i_slice] = spmat[i_slice_from, j_slice_from].trans()
        # else:
        #     out[j_slices[sj],i_slice] = spmat[j_slice_from, i_slice_from]

        # Slice tau.
        @deterministic
        def tau_offdiag(tau = self.tau):
            out = cvx.base.spmatrix([],[],[], (self.fixed_len, self.changeable_len))
            Nc = len(self.changeable_slices_from)
            Nf = len(self.fixed_slices_from)
            for i in xrange(Nc):
                sfi = self.changeable_slices_from[i]
                sti = self.changeable_slices_to[i]
                for j in xrange(1, Nf):
                    sfj = self.fixed_slices_from[j]
                    stj = self.fixed_slices_to[j]
                    if sfi.start < sfj.start:
                        out[stj, sti] = tau[sfi,sfj].trans()
                    else:
                        out[stj, sti] = tau[sfj, sfi]
            return out
            # return slice_by_stochastics(tau, self.changeable_stochastic_list,
            #     self.fixed_stochastic_list, self.slices, self.changeable_slices, self.stochastic_len, self.moral_neighbors)
        self.tau_offdiag = tau_offdiag

        # Condition canonical eta parameter.
        # Slice canonical eta parameter by changeable stochastics.
        @deterministic(cache_depth=2)
        def eta(full_eta = full_eta, x=x, tau_offdiag = tau_offdiag):
            eta = cvx.base.matrix(0.,size=(self.changeable_len, 1))
            for i in xrange(len(self.changeable_slices_from)):
                sfi = self.changeable_slices_from[i]
                sti = self.changeable_slices_to[i]
                eta[sti] = full_eta[sfi]
            # for s in self.changeable_stochastic_list:
            #     eta[self.changeable_slices[s]] = full_eta[self.slices[s]]
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

        changeable_slices_from = []
        fixed_slices_from = []
        for s in self.changeable_stochastic_list:
            changeable_slices_from.append(self.slices[self.stochastic_list_numbers[s]])
        for s in self.fixed_stochastic_list:
            fixed_slices_from.append(self.slices[self.stochastic_list_numbers[s]])

        self.changeable_slices_from, self.changeable_slices_to = \
            contiguize_slices(self.changeable_stochastic_list, changeable_slices_from, self.changeable_slices)
        self.fixed_slices_from, self.fixed_slices_to = \
            contiguize_slices(self.fixed_stochastic_list, fixed_slices_from, self.fixed_slices)

        @deterministic
        def changeable_tau_slice(tau = self.tau):
            out = cvx.base.spmatrix([],[],[], (self.changeable_len, self.changeable_len))
            N = len(self.changeable_slices_from)
            for i in xrange(N):
                sfi = self.changeable_slices_from[i]
                sti = self.changeable_slices_to[i]
                for j in xrange(i, N):
                    sfj = self.changeable_slices_from[j]
                    stj = self.changeable_slices_to[j]
                    out[sti, stj] = tau[sfi,sfj]
            return out
            # return slice_by_stochastics(tau, self.changeable_stochastic_list,
            #     self.changeable_stochastic_list, self.slices, self.changeable_slices, self.stochastic_len, self.moral_neighbors)
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
        assign_from_sparse(dev, self.changeable_stochastic_list, self.changeable_slices)


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

    # # =========================================
    # # = Test case 1: Some old smallish model. =
    # # =========================================
    # A = Normal('A',1,1)
    # B = Normal('B',A,2*np.ones(2))
    # C_tau = np.diag([.5,.5])
    # C_tau[0,1] = C_tau[1,0] = .25
    # C = MvNormal('C',B, C_tau,observed=True)
    # D_mean = LinearCombination('D_mean', x=[np.ones((3,2))], y=[C])
    #
    # D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    # # D = Normal('D',D_mean,.5*np.ones(3))
    # G = NormalSubmodel([B,C,A,D,D_mean])
    # # G = NormalSubmodel([A,B,C])
    # G.draw_conditional()
    #
    # dense_tau = sp_to_ar(G.tau.value)
    # for i in xrange(dense_tau.shape[0]):
    #     for j in xrange(i):
    #         dense_tau[i,j] = dense_tau[j,i]
    # CC=(dense_tau).I
    # sig_tau = np.linalg.cholesky(dense_tau)


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
    # x_list[N/2].observed=True
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
