from numpy.testing import *
from pymc.gp import *
from pymc.gp.cov_funs import *
from numpy import *
from copy import copy

from pymc import six
xrange = six.moves.xrange

random.seed(42)

def zero_fun(x):
    return zeros(x.shape[:-1], dtype=float)
M = Mean(zero_fun)

lin = linspace(-1., 1., 50)
x, y = meshgrid(lin, lin)
xstack = zeros(x.shape + (2,), dtype=float)
xstack[:, :, 0] = x
xstack[:, :, 1] = y

lin_obs = linspace(-.5, .5, 2)
xobs, yobs = meshgrid(lin_obs, lin_obs)
ostack = zeros(xobs.shape + (2,), dtype=float)
ostack[:, :, 0] = xobs
ostack[:, :, 1] = yobs

V = .02 * ones((2, 2), dtype=float)
data = ones((2, 2), dtype=float)
N = 10

coef_cov = ones((2 * N + 1, 2 * N + 1), dtype=float)
for i in xrange(1, 2 * N + 1):
    int_i = int(((i + 1) / 2))
    for j in xrange(1, 2 * N + 1):
        int_j = int(((j + 1) / 2))
        coef_cov[i, j] = 1. / (sqrt((int_i + int_j) ** 2)) ** 2.5

basis = fourier_basis([N, N])
C = SeparableBasisCovariance(basis, coef_cov, xmin=[-4., -.4], xmax=[4., 4.])


class test_basiscov(TestCase):

    def test_sep(self):

        observe(M, C, obs_mesh=ostack, obs_V=V, obs_vals=data)

    def test_nonsep(self):

        basis_array = zeros((2 * N + 1, 2 * N + 1), dtype=object)
        for i in arange(2 * N + 1):

            if i % 2 == 0:
                def funi(x, xmin, xmax, i=i):
                    T = xmax[0] - xmin[0]
                    return cos(i * pi * (x - xmin[0]) / T)

            else:
                def funi(x, xmin, xmax, i=i):
                    T = xmax[0] - xmin[0]
                    return sin((i + 1) * pi * (x - xmin[0]) / T)

            for j in arange(2 * N + 1):

                if j % 2 == 0:
                    def funj(x, xmin, xmax, j=j):
                        T = xmax[1] - xmin[1]
                        return cos(j * pi * (x - xmin[1]) / T)
                else:
                    def funj(x, xmin, xmax, j=j):
                        T = xmax[1] - xmin[1]
                        return sin((j + 1) * pi * (x - xmin[1]) / T)

                def fun_now(x, xmin, xmax, funi=funi, funj=funj):
                    return funi(x[:, 0], xmin, xmax) * \
                        funj(x[:, 1], xmin, xmax)

                basis_array[i, j] = fun_now

        C2 = BasisCovariance(
            basis_array, coef_cov, xmin=[-4., -.4], xmax=[4., 4.])
        assert_almost_equal(C2(xstack), C(xstack))

        observe(M, C, obs_mesh=ostack, obs_V=V, obs_vals=data)
