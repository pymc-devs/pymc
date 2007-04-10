__docformat__='reStructuredText'

from Covariance import Covariance
from Mean import Mean
from Realization import Realization
from PyMC2 import Parameter, Node, Container, parameter, node
from GPSamplingMethods import *
from GPutils import *

class GaussianProcess(Container):
    """
    G = GaussianProcess(cov_fun, mean_fun, cov_params, mean_params[, S_needed,
                        base_mesh, obs_mesh, obs_taus, lintrans, obs_vals, name])
    
    A special type of PyMC object container which holds a Gaussian process.
    
    Attributes:
    - f:    A Realization-valued Parameter.
    - M:    A Mean-valued Node.
    - C:    A Covariance-valued Node.
    - S:    A SamplingMethod for f, if S_needed = True.
    - name: Name.
    """
    def __init__(   self,
                    cov_fun, 
                    mean_fun,
                    cov_params,
                    mean_params,
                    S_needed = True,
                    base_mesh = None, 
                    obs_mesh = None,
                    obs_taus = None, 
                    lintrans = None, 
                    obs_vals = None,
                    name = 'GP'):
    
        # If the user wants to observe the GP, the parent structure is
        # more complicated than otherwise.
        
        if obs_mesh is not None:

            @node
            def C(  eval_fun = cov_fun, 
                    base_mesh = base_mesh, 
                    obs_taus = obs_taus, 
                    lintrans= lintrans,  
                    **cov_params):
                
                C = Covariance(eval_fun, base_mesh, **cov_params)
                observe_cov(C, obs_mesh, obs_taus, lintrans, obs_vals)
                return C
        
            @node
            def M(  eval_fun = mean_fun,
                    C=C, 
                    obs_vals = obs_vals,
                    **mean_params):
                
                M = Mean(eval_fun, C.base_mesh, **mean_params)
                observe_mean_from_cov(M, C, obs_vals)
                return M
        
        else:

            @node
            def C(eval_fun = cov_fun, base_mesh = base_mesh, **cov_params):
                return Covariance(eval_fun, base_mesh, **cov_params)
        
            @node
            def M(eval_fun = mean_fun, base_mesh = base_mesh, **mean_params):
                return Mean(eval_fun, base_mesh, **mean_params)
            
        @parameter
        def f(value=None, M=M, C=C):

            def logp(value, M, C):
                return GP_logp(value,M,C)
        
            def random(M, C):
                return Realization(M,C)
            
            rseed = 1.


        self.C = C
        self.M = M
        self.f = f
    
        PyMC_object_list = [C,M,f]
        
        if S_needed:
            self.S = GPMetropolis
            PyMC_object_list.append(self.S)
    
        Container.__init__(self, iterable=PyMC_object_list, name = name)