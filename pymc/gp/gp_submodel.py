# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

__all__ = ['GaussianProcess','GPEvaluation','GPSubmodel']

import pymc as pm
import copy
import numpy as np
from Realization import Realization


__all__ = ['GaussianProcess', 'GPEvaluation', 'GPSubmodel']

def gp_logp(x, M, C, mesh, f_eval):
    raise TypeError, 'GP objects have no logp function'

def gp_rand(M, C, mesh, f_eval, size=None):
    # M and C are input pre-observed, so no need to 
    out = pm.gp.Realization(M, C)
    out.x_sofar = mesh
    out.f_sofar = f_eval
    return out

class GaussianProcess(pm.Stochastic):
    
    def __init__(self,name,submodel,doc=None,trace=True,value=None,rseed=False,
                    observed=False,cache_depth=2,plot=None,verbose=None,isdata=None):
                
        self.submodel = submodel
        
        pm.Stochastic.__init__(self,gp_logp,doc,name,
                                {'M':submodel.M_obs, 'C':submodel.C_obs,'mesh':submodel.mesh, 'f_eval':submodel.f_eval},
                                gp_rand,trace,value,np.dtype('object'),rseed,observed,cache_depth,plot,verbose,isdata,False)
        
        self.rand()
        
    def gen_lazy_function(self):
        pass
        
    def get_logp(self):
        raise TypeError, 'Gaussian process %s has no logp.'%self.__name__
        
    def set_logp(self, new_logp):
        raise TypeError, 'Gaussian process %s has no logp.'%self.__name__
        
    logp = property(fget = get_logp, fset = set_logp)



def gp_eval_logp(x, mu, sig, piv):
    """
    Log-probability function for GP evaluations
    """
    return pm.mv_normal_chol_like(x[np.argsort(piv)],mu,sig)

def gp_eval_rand(mu,sig,piv):
    return pm.rmv_normal_chol(mu,sig)[piv]
    
GPEvaluation =  pm.stochastic_from_dist('g_p_evaluation', gp_eval_logp, gp_eval_rand, dtype=np.dtype('float'), mv=True)


    
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

        @pm.deterministic(trace=tally_all, name='%s_covariance_bits'%name)
        def covariance_bits(C=C,mesh=mesh):
            C_obs = copy.copy(C)
            try:
                U, relslice, offdiag = C_obs.observe(mesh, np.zeros(mesh.shape[0]), assume_full_rank=True, output_type='s')
                return U.T.copy('F'), relslice, np.asarray(np.dot(offdiag.T, offdiag), order='F'), C_obs
            except np.linalg.LinAlgError:
                return None

        S_eval, relslice, offdiag, C_obs = covariance_bits    
        
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
        
        f_eval = GPEvaluation('%s_f_eval'%name, mu=M_eval, sig=S_eval, piv=relslice, value=init_vals, trace=True)
        
        @dtrm(trace=tally_all, name='%s_covariance_bits'%name)
        def M_obs(M=M, f_eval=f_eval, C_obs=C_obs, mesh=mesh, relslice=relslice):
            M_obs = copy.copy(M)
            M_obs.observe(C_obs,mesh,relslice)
            return M_obs
        
        self.f_eval = f_eval
        f = GaussianProcess('%s_f'%name, self, trace=tally_all)
        pm.ObjectContainer.__init__(self, locals())
        
    def getobjects(self):
        names = ['M_eval','C_eval','S_eval','f_eval','f','fr_check']
        return dict(zip(['%s_%s'%(self.name, name) for name in names], [getattr(self, name) for name in names]))