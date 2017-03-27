from theano import theano, tensor as tt
from pymc3.variational.opvi import Operator
import pymc3 as pm

__all__ = [
    'KL'
]


class KL(Operator):
    """
    Operator based on Kullback Leibler Divergence
    .. math::

        KL[q(v)||p(v)] = \int q(v)\log\\frac{q(v)}{p(v)}dv
    """
    def apply(self, f):
        z = self.input
        return self.logq_norm(z) - self.logp_norm(z)

# SVGD Implementation


class KSD(Operator):
    """
    Kernelized Stein Discrepancy

    Input: A target distribution with density function :math:`p(x)`
        and a set of initial particles :math:`{x^0_i}^n_{i=1}`
    Output: A set of particles :math:`{x_i}^n_{i=1}` that approximates the target distribution.
    .. math::

        x_i^{l+1} \leftarrow \epsilon_l \hat{\phi}^{*}(x_i^l)
        \hat{\phi}^{*}(x) = \frac{1}{n}\sum^{n}_{j=1}[k(x^l_j,x) \nabla_{x^l_j} logp(x^l_j)+ \nabla_{x^l_j} k(x^l_j,x)]

    Parameters
    ----------
    approx : pm.Histogram

    References
    ----------
    - Qiang Liu, Dilin Wang (2016)
        Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm
        arXiv:1608.04471
    """
    NEED_F = True
    HISTOGRAM_BASED = True

    def __init__(self, approx):
        if not isinstance(approx, pm.Histogram):
            raise ValueError('approx should be a Histogram, got %r' % approx)
        Operator.__init__(self, approx)

    def apply(self, f):
        # f: kernel function for KSD f(histogram) -> (k(x,.), \nabla_x k(x,.))
        X = self.approx.histogram
        dlogpdx = theano.scan(
            fn=lambda z: theano.grad(self.logp(z), z),
            sequences=X
        )[0]    # bottleneck
        Kxy, dxkxy = f(X)
        svgd_grad = (tt.dot(Kxy, dlogpdx) + dxkxy) / X.shape[0].astype('float32')
        return [-1 * svgd_grad]
