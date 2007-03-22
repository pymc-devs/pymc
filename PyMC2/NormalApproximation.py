# FIXME: fmin_bfgs doesn't seem to be working. 
# FIXME: Possibly just write grad and hessian methods
# FIXME: and use one of the other optimizers.

# TODO: Make mu and C model nodes that lazily update
# TODO: when any of the model's parameters have changed.

# TODO: Provide methods for nice extraction of mu and C
# TODO: relevant to particular parameters. N.C[p1, p2]
# TODO: would be a nice syntax.

from PyMCObjects import Parameter, Node, PyMCBase
from Container import Container
from Model import Model
from numpy import zeros, inner, asmatrix, ndarray, reshape, shape
from numpy.random import normal
from utils import msqrt

try:
    from scipy.optimize import fmin_bfgs
except ImportError:
    raise ImportError, 'scipy must be installed to use NormalApproximation.'

class NormalApproximation(Model):

    def __init__(self, input, db='ram', eps=.000001):
        Model.__init__(self, input, db)

        # Allocate memory for internal traces and get parameter slices
        self._slices = {}
        self._len = 0
        self.param_len = {}
        
        for parameter in self.parameters:
            if isinstance(parameter.value, ndarray):
                self.param_len[parameter] = len(parameter.value.ravel())
            else:
                self.param_len[parameter] = 1
            self._slices[parameter] = slice(self._len, self._len + self.param_len[parameter])
            self._len += self.param_len[parameter]
        self.eps = eps
        
        self._len_range = arange(self._len)
        self.grad = zeros(self._len, dtype=float)
        self.hess = zeros((self._len, self._len), dtype=float)
        
        self._maximize()
        
    def _get_logp(self):
        return sum([p.logp for p in self.parameters]) + sum([p.logp for p in self.data])
    
    logp = property(_get_logp)
    
    def _maximize(self):
        p = zeros(self._len,dtype=float)
        for parameter in self.parameters:
            p[self._slices[parameter]] = parameter.value.ravel()

        # fmin_bfgs doesn't seem to be working.
        O = fmin_bfgs(f=self._get_logp_from_p, x0=p, fprime=self._get_gradient_from_p, full_output=True)
        self.mu = O[0]
        self.C = asmatrix(O[3])
        self._sig = msqrt(self.C).T
    
    def _set_parameters(self, p):
        for parameter in self.parameters:
            parameter.value = parameter.value + reshape(p[self._slices[parameter]],shape(parameter.value))
    
    def _get_logp_from_p(self, p):
        self._set_parameters(p)
        return -1.*self.logp
        
    def _get_gradient_from_p(self, p):
        # Try, may not work.
        
        self._set_parameters(p)
        
        for param in self.parameters:
            base_logp = self.logp
            for i in range(self.param_len[param]):
                val = param.value.ravel()
                h = val[i] * eps
                val[i] = val[i] + h
                param.value = reshape(val, shape(param.value))

                up_logp = self.logp
                self.grad[self._slices[parameter]][i] = (up_logp - base_logp) / h

                param.value = param.last_value

        return self.grad
        
    def draw(self):
        devs = normal(size=self._len)
        p = inner(devs, self._sig)
        self._set_parameters(p)