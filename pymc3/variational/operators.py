import warnings
import collections
from theano import tensor as tt
from pymc3.theanof import change_flags
from pymc3.variational.opvi import Operator, ObjectiveFunction
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
        return self.logq_norm - self.logp_norm

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

    def get_input(self):
        if hasattr(self.approx, 'histogram'):
            return self.approx.symbolic_random_local_matrix, self.approx.histogram
        else:
            return self.approx.symbolic_random_local_matrix, self.approx.symbolic_random_global_matrix

    @change_flags(compute_test_value='off')
    def __call__(self, nmc, **kwargs):
        op = self.op  # type: KSD
        grad = op.apply(self.tf)
        loc_size = self.approx.local_size
        local_grad = grad[..., :loc_size]
        global_grad = grad[..., loc_size:]
        if 'more_obj_params' in kwargs:
            params = self.obj_params + kwargs['more_obj_params']
        else:
            params = self.test_params + kwargs['more_tf_params']
            grad *= pm.floatX(-1)
        zl, zg = self.get_input()
        zl, zg, grad, local_grad, global_grad = self.approx.set_size_and_deterministic(
            (zl, zg, grad, local_grad, global_grad),
            nmc, 0)
        grad = tt.grad(None, params, known_grads=collections.OrderedDict([
            (zl, local_grad),
            (zg, global_grad)
        ]), disconnected_inputs='ignore')
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

    def get_input(self):
        if isinstance(self.approx, pm.Empirical):
            return self.approx.histogram
        else:
            return self.approx.symbolic_random_total_matrix

    def apply(self, f):
        # f: kernel function for KSD f(histogram) -> (k(x,.), \nabla_x k(x,.))
        input_matrix = self.get_input()
        stein = Stein(
            approx=self.approx,
            kernel=f,
            input_matrix=input_matrix,
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
