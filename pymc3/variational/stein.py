from theano import theano, tensor as tt
from pymc3.variational.opvi import node_property
from pymc3.variational.test_functions import rbf
from pymc3.theanof import memoize, floatX, change_flags

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
        self.input_matrix = input_matrix

    @node_property
    def grad(self):
        n = floatX(self.input_matrix.shape[0])
        temperature = self.temperature
        svgd_grad = (self.density_part_grad / temperature +
                     self.repulsive_part_grad)
        return svgd_grad / n

    @node_property
    def density_part_grad(self):
        Kxy = self.Kxy
        dlogpdx = self.dlogp
        return tt.dot(Kxy, dlogpdx)

    @node_property
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

    @node_property
    def logp_norm(self):
        return self.approx.sized_symbolic_logp / self.approx.normalizing_constant

    @node_property
    def dlogp(self):
        loc_random = self.input_matrix[..., :self.approx.local_size]
        glob_random = self.input_matrix[..., self.approx.local_size:]
        loc_grad, glob_grad = tt.grad(
            self.logp_norm.sum(),
            [self.approx.symbolic_random_local_matrix,
             self.approx.symbolic_random_global_matrix],
            disconnected_inputs='ignore'
        )
        loc_grad, glob_grad = theano.clone(
            [loc_grad, glob_grad],
            {self.approx.symbolic_random_local_matrix: loc_random,
             self.approx.symbolic_random_global_matrix: glob_random}
        )
        return tt.concatenate([loc_grad, glob_grad], axis=-1)

    @memoize
    @change_flags(compute_test_value='off')
    def _kernel(self):
        return self._kernel_f(self.input_matrix)
