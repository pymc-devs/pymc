from theano import theano, tensor as tt
from pymc3.variational.test_functions import rbf
from pymc3.theanof import memoize, floatX

__all__ = [
    'Stein'
]


class Stein(object):
    def __init__(self, approx, kernel=rbf, input_matrix=None, temperature=1):
        self.approx = approx
        self.temperature = floatX(temperature)
        self._kernel_f = kernel
        if input_matrix is None:
            input_matrix = tt.matrix('stein_input_matrix')
            input_matrix.tag.test_value = approx.random(10).tag.test_value
        self.input_matrix = input_matrix

    @property
    @memoize
    def grad(self):
        n = floatX(self.input_matrix.shape[0])
        temperature = self.temperature
        svgd_grad = (self.density_part_grad / temperature +
                     self.repulsive_part_grad)
        return svgd_grad / n

    @property
    @memoize
    def density_part_grad(self):
        Kxy = self.Kxy
        dlogpdx = self.dlogp
        return tt.dot(Kxy, dlogpdx)

    @property
    @memoize
    def repulsive_part_grad(self):
        t = self.approx.normalizing_constant
        dxkxy = self.dxkxy
        return dxkxy/t

    @property
    def Kxy(self):
        return self._kernel()[0]

    @property
    def dxkxy(self):
        return self._kernel()[1]

    @property
    @memoize
    def logp(self):
        return theano.scan(
            fn=lambda zg: self.approx.logp_norm(zg),
            sequences=[self.input_matrix]
        )[0]

    @property
    @memoize
    def dlogp(self):
        return theano.scan(
            fn=lambda zg: theano.grad(self.approx.logp_norm(zg), zg),
            sequences=[self.input_matrix]
        )[0]

    @memoize
    def _kernel(self):
        return self._kernel_f(self.input_matrix)

    def get_approx_input(self, size=100):
        """

        Parameters
        ----------
        size : if approx is not Empirical, takes `n=size` random samples

        Returns
        -------
        matrix
        """
        if hasattr(self.approx, 'histogram'):
            return self.approx.histogram
        else:
            return self.approx.random(size)
