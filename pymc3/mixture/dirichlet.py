import numpy as np
import numpy.random as nr
import theano.tensor as tt
from scipy import stats

from .arraystep import ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars
from ..distributions import draw_values

__all__ = ['Dirichlet']

class StickBreakingProcess(Continuous):
    def __init__(self, alpha, shape=None, *args, **kwargs):
        super(StickBreaking, self).__init__(*args, shape=shape, **kwargs)
        self.shape = tt.as_tensor_variable(shape)
        self.alpha = tt.as_tensor_variable(alpha)
        
    def logp(self, value):
        raise NotImplementedError
        
