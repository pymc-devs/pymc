from theano import function
import theano.tensor as tt

import pymc3.distributions.special as ps
import scipy.special as ss
import numpy as np

from .checks import close_to


def test_functions():
    xvals = list(map(np.atleast_1d, [.01, .1, 2, 100, 10000]))

    x = tt.dvector('x')
    x.tag.test_value = xvals[0]

    p = tt.iscalar('p')
    p.tag.test_value = 1

    gammaln = function([x], ps.gammaln(x))
    psi = function([x], ps.psi(x))
    function([x, p], ps.multigammaln(x, p))
    for x in xvals:
        yield check_vals, gammaln, ss.gammaln, x
    for x in xvals[1:]:
        yield check_vals, psi, ss.psi, x

"""
scipy.special.multigammaln gives bad values if you pass a non scalar to a
In [14]:

    import scipy.special
    scipy.special.multigammaln([2.1], 3)
    Out[14]:
        array([ 1.76253257,  1.60450306,  1.66722239])
"""


def t_multigamma():
    xvals = list(map(np.atleast_1d, [0, .1, 2, 100]))

    x = tt.dvector('x')
    x.tag.test_value = xvals[0]

    p = tt.iscalar('p')
    p.tag.test_value = 1

    multigammaln = function([x, p], ps.multigammaln(x, p))

    def ssmultigammaln(a, b):
        return ss.multigammaln(a[0], b)

    for p in [0, 1, 2, 3, 4, 100]:
        for x in xvals:
            yield check_vals, multigammaln, ssmultigammaln, x, p


def check_vals(fn1, fn2, *args):
    v = fn1(*args)
    close_to(v, fn2(*args), 1e-6)
