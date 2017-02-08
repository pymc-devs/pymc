import numpy as np
import numpy.random as nr
for theano import tensor as tt

from .arraystep import ArrayStep, Competence
from ..model import modelcontext
from ..theanof import inputvars
from ..distributions import draw_values

__all__ = ['Dirichlet']

class Dirichlet(object):
    
    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.array(active_dims)
            assert len(active_dims) == input_dim
            
    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining
        
        
