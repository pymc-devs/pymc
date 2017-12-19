from theano import theano, tensor as tt
from pymc3.variational.opvi import node_property
from pymc3.variational.test_functions import rbf
from pymc3.theanof import floatX, change_flags
from pymc3.memoize import WithMemoization, memoize

__all__ = [
    'Stein'
]


class Stein(WithMemoization):
    def __init__(self, approx, kernel=rbf, use_histogram=True, temperature=1):
        self.approx = approx
        self.temperature = floatX(temperature)
        self._kernel_f = kernel
        self.use_histogram = use_histogram

    @property
    def input_joint_matrix(self):
        if self.use_histogram:
            return self.approx.joint_histogram
        else:
            return self.approx.symbolic_random

    @node_property
    def approx_symbolic_matrices(self):
        if self.use_histogram:
            return self.approx.collect('histogram')
        else:
            return self.approx.symbolic_randoms

    @node_property
    def dlogp(self):
        grad = tt.grad(
            self.logp_norm.sum(),
            self.approx_symbolic_matrices
        )

        def flatten2(tensor):
            return tensor.flatten(2)
        return tt.concatenate(list(map(flatten2, grad)), -1)

    @node_property
    def grad(self):
        n = floatX(self.input_joint_matrix.shape[0])
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
        t = self.approx.symbolic_normalizing_constant
        dxkxy = self.dxkxy
        return dxkxy / t

    @property
    def Kxy(self):
        return self._kernel()[0]

    @property
    def dxkxy(self):
        return self._kernel()[1]

    @node_property
    def logp_norm(self):
        sized_symbolic_logp = self.approx.sized_symbolic_logp
        if self.use_histogram:
            sized_symbolic_logp = theano.clone(
                sized_symbolic_logp,
                dict(zip(self.approx.symbolic_randoms, self.approx.collect('histogram')))
            )
        return sized_symbolic_logp / self.approx.symbolic_normalizing_constant

    @memoize
    @change_flags(compute_test_value='off')
    def _kernel(self):
        return self._kernel_f(self.input_joint_matrix)
