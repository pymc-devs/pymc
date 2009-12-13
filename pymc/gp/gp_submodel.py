# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

__all__ = ['GaussianProcess','GPEvaluation','GPSubmodel']

import pymc as pm
import copy
import numpy as np
from Realization import Realization


__all__ = ['GaussianProcess', 'GPEvaluation', 'GPSubmodel']

def gp_logp(x, M, C, mesh, f_eval, M_obs, C_obs):
    raise TypeError, 'GP objects have no logp function'

def gp_rand(M, C, mesh, f_eval, M_obs, C_obs, size=None):
    # M and C are input pre-observed, so no need to 
    out = pm.gp.Realization(M_obs, C_obs)
    out.x_sofar = mesh
    out.f_sofar = f_eval
    out.M = M
    out.C = C
    return out

class GaussianProcess(pm.Stochastic):
    
    def __init__(self,name,submodel,doc=None,trace=True,value=None,rseed=False,
                    observed=False,cache_depth=2,plot=None,verbose=None,isdata=None):
                
        self.submodel = submodel
        
        pm.Stochastic.__init__(self,gp_logp,doc,name,
                                {'M':submodel.M, 'C':submodel.C,'mesh':submodel.mesh, 'f_eval':submodel.f_eval, 'M_obs':submodel.M_obs, 'C_obs':submodel.C_obs},
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
        
        self.M = M
        self.C = C
        self.mesh = mesh
        self.name = name

        @pm.deterministic(trace=tally_all, name='%s_covariance_bits'%name)
        def covariance_bits(C=C,mesh=mesh):
            """
            This is kind of complicated. Both the realization 'f' and the on-mesh evaluation
            'f_eval' need the Cholesky factor of the covariance evaluation. The Gibbs step
            method also needs the full covariance evaluation. 
            
            Both these things can be got as byproducts of Covariance.observe. Keeping the
            observed covariance and using it as the parent of f means the computations only
            get done once.
            """
            C_obs = copy.copy(C)
            try:
                # Note: no pivot is returned because assume_full_rank=True.
                U, C_eval = C_obs.observe(mesh, np.zeros(mesh.shape[0]), assume_full_rank=True, output_type='s')
                return U.T.copy('F'), C_eval, C_obs
            except np.linalg.LinAlgError:
                return None

        S_eval, C_eval, C_obs = covariance_bits    
        
        M_eval = pm.Lambda('%s_M_eval'%name, lambda M=M, mesh=mesh: M(mesh), trace=tally_all)
                
        @pm.potential(name = '%s_fr_check'%name)
        def fr_check(cb=covariance_bits):
            """
            Forbids non-positive-definite C_evals.
            """
            if cb is None:
                return -np.inf
            else:
                return 0
        
        f_eval = GPEvaluation('%s_f_eval'%name, mu=M_eval, sig=S_eval, value=init_vals, trace=True)
        
        @dtrm(trace=tally_all, name='%s_covariance_bits'%name)
        def M_obs(M=M, f_eval=f_eval, C_obs=C_obs, mesh=mesh):
            """
            Creates an observed mean object to match C_obs.
            """
            M_obs = copy.copy(M)
            M_obs.observe(C_obs,mesh,f_eval)
            return M_obs
        
        self.f_eval = f_eval
        f = GaussianProcess('%s_f'%name, self, trace=tally_all)
        pm.ObjectContainer.__init__(self, locals())
        
    def getobjects(self):
        names = ['M_eval','C_eval','S_eval','f_eval','f','fr_check']
        return dict(zip(['%s_%s'%(self.name, name) for name in names], [getattr(self, name) for name in names]))