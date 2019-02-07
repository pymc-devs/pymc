import theano
import theano.tensor as tt
import numpy as np
from .distribution import NoDistribution, Distribution

__all__ = ['Simulator']


class Simulator(NoDistribution):

    def __init__(self, function, *args, **kwargs):

        self.function = function
        observed = self.data
        super(
            Simulator,
            self).__init__(
            shape=np.prod(
                observed.shape),
            dtype=observed.dtype,
            *args,
            **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Simulator

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """

        print('Not implemented yet')

    def logp(self, value):
        """


        Parameters
        ----------
        value : numeric
            Value for which log-probability is calculated.
        Returns
        -------
        TensorVariable
        """
        return tt.zeros_like(value)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r'\text{%s}' % name
        function = dist.function
        params = dist.parameters
        sum_stat = dist.sum_stat
        return r'${} \sim \text{{Simulator}}(\mathit{{function}}={},~\mathit{{parameters}}={},~\mathit{{summary statistics}}={})$'.format(
            name, function, params, sum_stat)
