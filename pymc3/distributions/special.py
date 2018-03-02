import numpy as np
import theano.tensor as tt
from theano.scalar.basic_scipy import GammaLn, Psi
from theano import scalar

__all__ = ['gammaln', 'multigammaln', 'psi', 'log_i0']

scalar_gammaln = GammaLn(scalar.upgrade_to_float, name='scalar_gammaln')
gammaln = tt.Elemwise(scalar_gammaln, name='gammaln')


def multigammaln(a, p):
    """Multivariate Log Gamma

    Parameters
    ----------
    a : tensor like
    p : int
       degrees of freedom. p > 0
    """
    i = tt.arange(1, p + 1)
    return (p * (p - 1) * tt.log(np.pi) / 4.
            + tt.sum(gammaln(a + (1. - i) / 2.), axis=0))


def log_i0(x):
    """
    Calculates the logarithm of the 0 order modified Bessel function of the first kind""
    """
    return tt.switch(tt.lt(x, 5), tt.log1p(x**2. / 4. + x**4. / 64. + x**6. / 2304.
                                           + x**8. / 147456. + x**10. / 14745600.
                                           + x**12. / 2123366400.),
                                  x - 0.5 * tt.log(2. * np.pi * x) + tt.log1p(1. / (8. * x)
                                  + 9. / (128. * x**2.) + 225. / (3072. * x**3.)
                                  + 11025. / (98304. * x**4.)))


scalar_psi = Psi(scalar.upgrade_to_float, name='scalar_psi')
psi = tt.Elemwise(scalar_psi, name='psi')
