# TODO: Give C, mu and V attributes to NormalSubmodel, make NormalModel a very thin wrapper in NormalSubModel.py.

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

import pymc
import numpy as np
from NormalSubmodel import NormalSubmodel, cvx, sp_to_ar

__all__ = ['NormalModel', 'NormalModelMu', 'NormalModelC']


class NormalModelMu(object):
    """
    Returns the mean vector of some variables.

    Usage: If p1 and p2 are array-valued stochastic variables and N is a
    NormalModel object,

    N.mu(p1,p2)

    will give the approximate posterior mean of the ravelled, concatenated
    values of p1 and p2.
    """
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, *stochastics):

        tot_len = 0

        try:
            for p in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics

        for p in stochastic_tuple:
            tot_len += self.owner.NSM.stochastic_len[p]

        mu = np.empty(tot_len, dtype=float)

        start_index = 0

        for p in stochastic_tuple:
            this_len = self.owner.NSM.stochastic_len[p]
            mu[start_index:(start_index + this_len)] = \
                self.owner.NSM.changeable_mean.value[self.owner.NSM.changeable_slices[p]]
            start_index += this_len

        return mu


class NormalModelC(object):
    """
    Returns the covariance matrix of some variables.

    Usage: If p1 and p2 are array-valued stochastic variables and N is a
    NormalModel object,

    N.C(p1,p2)

    will give the approximate covariance matrix of the ravelled, concatenated
    values of p1 and p2
    """
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, *stochastics):

        try:
            for s in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics

        # Make list of slices corresponding to each Stochastic
        slices = []
        start = 0
        for s in stochastic_tuple:
            ls = self.owner.NSM.stochastic_len[s]
            slices.append(slice(start, start + ls))
            start += ls
        tot_len = start

        # Make identity matrix with zero rows inserted for variables in NSM.changeable_stochastics
        # but not in stochastics.
        partial_identity = cvx.base.matrix(0., (self.owner.NSM.changeable_len, tot_len))
        for i in xrange(len(stochastic_tuple)):

            from_slice = self.owner.NSM.changeable_slices[stochastic_tuple[i]]
            li = self.owner.NSM.stochastic_len[stochastic_tuple[i]]

            partial_identity[from_slice,slices[i]] = cvx.base.spmatrix(np.ones(li), xrange(li), xrange(li))

        # Backsolve against partial identity matrix
        cvx_cov = self.owner.NSM.backsolver.value(partial_identity, squared=True)

        # Slice appropriate rows from partial identity matrix.
        C = np.asmatrix(np.empty((tot_len,tot_len), dtype=float))
        for i in xrange(len(stochastic_tuple)):
            C[slices[i],:] = cvx_cov[self.owner.NSM.changeable_slices[stochastic_tuple[i]],:]

        return C

class NormalModel(pymc.Sampler):
    """
    N = NoralModel(input, db='ram', **kwds))

    A Sampler subclass whose variables comprise a Gaussian submodel.

    Useful attributes (after fit() is called):
    -  mu[p1, p2, ...]:    Returns the posterior mean vector of stochastic variables p1, p2, ...
    -  C[p1, p2, ...]:     Returns the posterior covariance of stochastic variables p1, p2, ...

    """
    def __init__(self, input, db='ram', **kwds):

        if not isinstance(input, NormalSubmodel):
            raise ValueError, 'input argument must be NormalSubmodel instance.'

        self.NSM = input
        pymc.Sampler.__init__(self, input, db=db, reinit_model=True, **kwds)

        self.mu = NormalModelMu(self)
        self.C = NormalModelC(self)

    def draw(self):
        self.NSM.draw_conditional()

    def __getattr__(self, attr):
        try:
            return object.__getattr__(self, attr)
        except:
            return getattr(self.NSM, attr)

    def __setattr__(self, attr, newval):
        try:
            object.__setattr__(self, attr, newval)
        except:
            setattr(self.NSM, attr, newval)

if __name__ == '__main__':
    from pylab import *

    import numpy as np
    from pymc import *

    # # =========================================
    # # = Test case 1: Some old smallish model. =
    # # =========================================
    # A = Normal('A',1,1)
    # B = Normal('B',A,2*np.ones(2))
    # C_tau = np.diag([.5,.5])
    # C_tau[0,1] = C_tau[1,0] = .25
    # C = MvNormal('C',B, C_tau, observed=True)
    # D_mean = LinearCombination('D_mean', x=[np.ones((3,2))], y=[C])
    #
    # D = MvNormal('D',D_mean,np.diag(.5*np.ones(3)))
    # # D = Normal('D',D_mean,.5*np.ones(3))
    # G = NormalSubmodel([B,C,A,D,D_mean])
    #
    # N = NormalModel(G)

    # ================================
    # = Test case 2: Autoregression. =
    # ================================


    N=100
    W = Uninformative('W',np.eye(2)*N)
    base_mu = Uninformative('base_mu', np.ones(2)*3)
    # W[0,1] = W[1,0] = .5
    x_list = [MvNormal('x_0',base_mu,W,value=np.zeros(2))]
    for i in xrange(1,N):
        # L = LinearCombination('L', x=[x_list[i-1]], y = [np.eye(2)])
        x_list.append(MvNormal('x_%i'%i,x_list[i-1],W))

    # W = N
    # x_list = [Normal('x_0',1.,W,value=0)]
    # for i in xrange(1,N):
    #     # L = LinearCombination('L', x=[x_list[i-1]], coefs = {x_list[i-1]:np.ones((2,2))}, offset=0)
    #     x_list.append(Normal('x_%i'%i,x_list[i-1],W))

    data_index = 2*N/3

    x_list[data_index].value = array([4.,-4.])
    x_list[data_index].observed=True

    G = NormalSubmodel(x_list)
    x_list.pop(data_index)

    N = NormalModel(G)
    close('all')
    figure()
    subplot(1,2,1)
    contourf(N.C[x_list][::2,::2].view(ndarray))
    subplot(1,2,2)
    plot(N.mu[x_list][::2])
    plot(N.mu[x_list][1::2])
