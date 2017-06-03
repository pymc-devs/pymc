import warnings
from theano import theano, tensor as tt
from pymc3.variational.opvi import Operator, ObjectiveFunction, _warn_not_used
from pymc3.variational.stein import Stein
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
    """Helper class for construction loss and updates for variational inference

    Parameters
    ----------
    op : :class:`KSD`
        OPVI Functional operator
    tf : :class:`TestFunction`
        OPVI TestFunction
    """

    def __init__(self, op, tf):
        if not isinstance(op, KSD):
            raise TypeError('Op should be KSD')
        ObjectiveFunction.__init__(self, op, tf)

    def get_input(self, n_mc):
        if hasattr(self.approx, 'histogram'):
            if n_mc is not None:
                _warn_not_used('n_mc', self.op)
            return self.approx.histogram
        elif n_mc is not None and n_mc > 1:
            return self.approx.random(n_mc)
        else:
            raise ValueError('Variational type approximation requires '
                             'sample size (`n_mc` : int > 1 should be passed)')

    def __call__(self, z, **kwargs):
        op = self.op  # type: KSD
        grad = op.apply(self.tf)
        if 'more_obj_params' in kwargs:
            params = self.obj_params + kwargs['more_obj_params']
        else:
            params = self.test_params + kwargs['more_tf_params']
            grad *= pm.floatX(-1)
        grad = theano.clone(grad, {op.input_matrix: z})
        grad = tt.grad(None, params, known_grads={z: grad})
        return grad


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

    def __init__(self, approx, temperature=1):
        Operator.__init__(self, approx)
        self.temperature = temperature
        self.input_matrix = tt.matrix('KSD input matrix')

    def apply(self, f):
        # f: kernel function for KSD f(histogram) -> (k(x,.), \nabla_x k(x,.))
        stein = Stein(
            approx=self.approx,
            kernel=f,
            input_matrix=self.input_matrix,
            temperature=self.temperature)
        return pm.floatX(-1) * stein.grad


class AKSD(KSD):
    def __init__(self, approx, temperature=1):
        warnings.warn('You are using experimental inference Operator. '
                      'It requires careful choice of temperature, default is 1. '
                      'Default temperature works well for low dimensional problems and '
                      'for significant `n_obj_mc`. Temperature > 1 gives more exploration '
                      'power to algorithm, < 1 leads to undesirable results. Please take '
                      'it in account when looking at inference result. Posterior variance '
                      'is often **underestimated** when using temperature = 1.', stacklevel=2)
        super(AKSD, self).__init__(approx, temperature)
    SUPPORT_AEVB = True
