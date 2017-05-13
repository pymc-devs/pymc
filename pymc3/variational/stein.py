from theano import theano, tensor as tt
from pymc3.variational.test_functions import rbf
from pymc3.theanof import memoize


class Stein(object):
    def __init__(self, approx, kernel=rbf, input_matrix=None):
        self.approx = approx
        self._kernel_f = kernel
        if input_matrix is None:
            input_matrix = tt.matrix('stein_input_matrix')
            input_matrix.tag.test_value = approx.random(10).tag.test_value
        self.input_matrix = input_matrix

    logp = property(lambda self: self.approx.input_matrix_logp)

    @property
    @memoize
    def grad(self):
        t = self.approx.normalizing_constant
        Kxy, dxkxy = self.Kxy, self.dxkxy
        dlogpdx = self.dlogp
        dxkxy /= t
        n = self.input_matrix.shape[0].astype('float32') / t
        svgd_grad = (tt.dot(Kxy, dlogpdx) + dxkxy) / n
        return svgd_grad

    @property
    def Kxy(self):
        return self._kernel()[0]

    @property
    def dxkxy(self):
        return self._kernel()[1]

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
