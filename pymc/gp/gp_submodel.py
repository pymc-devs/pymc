# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

# __all__ = ['GP_array_random', 'GP', 'GPMetropolis', 'GPNormal', 'GPParentMetropolis']

# FIXME: New pickle protocol for Realization. Just pickle the observation mesh and values, and maybe nugget or whatever.

import pymc as pm
import copy
import numpy as np
from Realization import Realization


__all__ = ['GaussianProcess', 'GPEvaluation', 'GPSubmodel']

def gp_logp(x, M, C, mesh, f_eval):
    raise TypeError, 'GP objects have no logp function'

def gp_rand(M, C, mesh, f_eval, size=None):
    return pm.gp.Realization(M, C, mesh, f_eval)

class GaussianProcess(pm.Stochastic):
    
    def __init__(self,name,submodel,doc=None,trace=True,value=None,rseed=False,
                    observed=False,cache_depth=2,plot=None,verbose=None,isdata=None):
                
        self.submodel = submodel
        
        pm.Stochastic.__init__(self,gp_logp,doc,name,
                                {'M':submodel.M, 'C':submodel.C,'mesh':submodel.mesh, 'f_eval':submodel.f_eval},
                                gp_rand,trace,value,np.dtype('object'),rseed,observed,cache_depth,plot,verbose,isdata,False)
        
        self.rand()
        
    def gen_lazy_function(self):
        pass
        
    def get_logp(self):
        raise TypeError, 'Gaussian process %s has no logp.'%self.__name__
        
    def set_logp(self, new_logp):
        raise TypeError, 'Gaussian process %s has no logp.'%self.__name__
        
    logp = property(fget = get_logp, fset = set_logp)


class GPEvaluation(pm.MvNormalChol):
    pass

    
class GPSubmodel(pm.ObjectContainer):
    """
    f = GPSubmodel('f', M, C, mesh, [obs_values, obs_nugget, tally_all])

    A stochastic variable valued as a Realization object.

    :Arguments:
        - M: A Mean instance or pm.deterministic variable valued as a Mean instance.
        - C: A Covariance instance or pm.deterministic variable valued as a Covariance instance.
        - mesh: The mesh on which self's log-probability will be computed. See documentation.
        - tally_all: By default, only f_eval and f are tallied. Turn this on to tally all.

    :SeeAlso: Realization, GPMetropolis, GPNormal
    """
    
    def __init__(self, name, M, C, mesh, init_vals=None, tally_all=False):
        
        # FIXME: Use S_eval for both logp and observation, meaning get it into f somehow.
        self.M = M
        self.C = C
        self.mesh = mesh
        self.name = name
        M_eval = pm.Lambda('%s_M_eval'%name, lambda M=M, mesh=mesh: M(mesh), trace=tally_all)
        C_eval = pm.Lambda('%s_C_eval'%name, lambda C=C, mesh=mesh: C(mesh,mesh), trace=tally_all)

        @pm.deterministic(name='%s_S_eval'%name, trace=tally_all)
        def S_eval(C_eval=C_eval):
            """
            Returns the Cholesky factor of C_eval if it is positive definite, or None if not.
            """
            try:
                return np.linalg.cholesky(C_eval)
            except np.linalg.LinAlgError:
                return None
                
        @pm.potential(name = '%s_fr_check'%name)
        def fr_check(S_eval=S_eval):
            """
            Forbids non-positive-definite C_evals.
            """
            if S_eval is None:
                return -np.inf
            else:
                return 0
        
        S_eval = pm.Lambda('%s_S_eval'%name, lambda C_eval=C_eval: np.linalg.cholesky(C_eval), trace=tally_all)
        f_eval = GPEvaluation('%s_f_eval'%name, mu=M_eval, sig=S_eval, value=init_vals, trace=True)
        self.f_eval = f_eval
        f = GaussianProcess('%s_f'%name, self, trace=tally_all)
        pm.ObjectContainer.__init__(self, locals())
        
    def getobjects(self):
        names = ['M_eval','C_eval','S_eval','f_eval','f','fr_check']
        return dict(zip(['%s_%s'%(self.name, name) for name in names], [getattr(self, name) for name in names]))