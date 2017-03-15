from pymc3.variational.opvi import Operator

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
        return self.logq(z) - self.logp(z)
