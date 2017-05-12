from theano import theano, tensor as tt
from pymc3.variational.opvi import Operator, ObjectiveFunction, _warn_not_used
from pymc3.variational.updates import adam
import pymc3 as pm

__all__ = [
    'KL',
    'KSD'
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


class KSDObjective(ObjectiveFunction):
    def add_obj_updates(self, updates, obj_n_mc=None, obj_optimizer=adam,
                        more_obj_params=None, more_replacements=None):
        if obj_n_mc is not None:
            _warn_not_used('obj_n_mc', self.op)
        d_obj_padams = self(None)
        d_obj_padams = theano.clone(d_obj_padams, more_replacements, strict=False)
        updates.update(obj_optimizer([d_obj_padams], self.obj_params))

    def __call__(self, z):
        return self.op.apply(self.tf)


class KSD(Operator):
    R"""
    Operator based on Kernelized Stein Discrepancy

    Input: A target distribution with density function :math:`p(x)`
        and a set of initial particles :math:`\{x^0_i\}^n_{i=1}`

    Output: A set of particles :math:`\{x_i\}^n_{i=1}` that approximates the target distribution.
    
    .. math::

        x_i^{l+1} \leftarrow \epsilon_l \hat{\phi}^{*}(x_i^l) \\
        \hat{\phi}^{*}(x) = \frac{1}{n}\sum^{n}_{j=1}[k(x^l_j,x) \nabla_{x^l_j} logp(x^l_j)+ \nabla_{x^l_j} k(x^l_j,x)]

    Parameters
    ----------
    approx : :class:`Empirical`
        Empirical Approximation used for inference

    References
    ----------
    -   Qiang Liu, Dilin Wang (2016)
        Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm
        arXiv:1608.04471
    """
    HAS_TEST_FUNCTION = True
    RETURNS_LOSS = False
    SUPPORT_AEVB = False
    OBJECTIVE = KSDObjective

    def __init__(self, approx):
        if not isinstance(approx, pm.Empirical):
            raise ValueError('approx should be an Empirical approximation, got %r' % approx)
        Operator.__init__(self, approx)

    def apply(self, f):
        # f: kernel function for KSD f(histogram) -> (k(x,.), \nabla_x k(x,.))
        X = self.approx.histogram
        t = self.approx.normalizing_constant
        dlogpdx = theano.scan(
            fn=lambda zg: theano.grad(self.logp_norm(zg), zg),
            sequences=[X]
        )[0]    # bottleneck
        Kxy, dxkxy = f(X)
        # scaling factor
        # not needed for Kxy as we already scaled dlogpdx
        dxkxy /= t
        n = X.shape[0].astype('float32') / t
        svgd_grad = (tt.dot(Kxy, dlogpdx) + dxkxy) / n
        return -1 * svgd_grad   # gradient
