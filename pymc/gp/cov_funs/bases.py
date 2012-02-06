# Copyright (c) Anand Patil, 2007

from numpy import *

from pymc import six
xrange = six.moves.xrange

__all__ = ['fourier_basis']

def fourier_basis(n):

    n_dim = len(n)
    basis = []

    for i in xrange(n_dim):

        def fun(x, xmin, xmax):
            return ones(x.shape[0], dtype=float)

        basis_now = [fun]

        for j in xrange(1,n[i]+1):

            def fun(x, xmin, xmax, j=j, i=i):
                T = xmax[i] - xmin[i]
                return cos(2.*j*pi*(x[:,i] - xmin[i]) / T)
            basis_now.append(fun)

            def fun(x, xmin, xmax, j=j, i=i):
                T = xmax[i] - xmin[i]
                return sin(2.*j*pi*(x[:,i] - xmin[i]) / T)
            basis_now.append(fun)

        basis.append(basis_now)

    return basis
