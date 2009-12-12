# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

# __all__ = ['GP_array_random', 'GP', 'GPMetropolis', 'GPNormal', 'GPParentMetropolis']

# FIXME: New pickle protocol for Realization. Just pickle the observation mesh and values, and maybe nugget or whatever. Be sure to delay the linear algebra until the last possible minute!

import pymc as pm
import linalg_utils
import copy
import types
import numpy as np
from gp_submodel import *

from Realization import Realization
from Mean import Mean
from Covariance import Covariance
from GPutils import observe, regularize_array

__all__ = ['GPEvaluationMetropolis', 'GPEvaluationGibbs', 'GPEvaluationAM', 'GPParentMetropolis']

class GPEvaluationMetropolis(pm.Metropolis):
    """
    Base class.
    """
    def __init__(self, submod, *args, **kwargs):        

        self.f_eval = submod.f_eval
        self.f = submod.f
        
        # Initialize superclass
        pm.StepMethod.__init__(self, [self.f, self.f_eval], *args, **kwargs)
                
        # Remove f from the set that will be used to compute logp_plus_loglike.
        self.markov_blanket_no_f = copy.copy(self.markov_blanket)
        self.markov_blanket_no_f.remove(self.f)
        
    def get_logp_plus_loglike(self):
        return pm.logp_of_set(self.markov_blanket_no_f)
    logp_plus_loglike = property(get_logp_plus_loglike)
        
    def reject(self):
        self.rejected += 1
        # Revert the field evaluation and the rest of the field.
        self.f_eval.revert()
        self.f.revert()

class GPEvaluationGibbs(GPEvaluationMetropolis):
    """
    Updates a GP evaluation f_eval. Assumes the only children of f_eval
    are as distributed follows:
    
    eps_p_f ~ Normal(f_eval[ti], 1./V)
    
    or
    
    eps_p_f ~ Normal(f_eval, 1./V)
    
    if ti is None.
    """
    def __init__(self, submod, V, eps_p_f, ti=None, tally=True, verbose=0):        

        # Initialize superclass
        GPEvaluationMetropolis.__init__(self, submod, tally=tally)
        
        self.V = V
        self.C_eval = self.f_eval.parents['C']
        self.M_eval = self.f_eval.parents['mu']
        self.eps_p_f = eps_p_f

        M_eval_shape = pm.utils.value(self.M_eval).shape
        C_eval_shape = pm.utils.value(self.C_eval).shape
        self.ti = ti or np.arange(M_eval_shape[0])

        # Work arrays
        self.scratch1 = np.asmatrix(np.empty(C_eval_shape, order='F'))
        self.scratch2 = np.asmatrix(np.empty(C_eval_shape, order='F'))
        self.scratch3 = np.empty(M_eval_shape)    

        # Initialize hidden attributes
        self.accepted = 0.
        self.rejected = 0.
        self._state = ['rejected', 'accepted', 'proposal_distribution']
        self._tuning_info = []
        self.proposal_distribution=None
    
    def tune(self):
        return False
            
    def propose(self):

        fc = pm.gp.fast_matrix_copy

        eps_p_f = pm.utils.value(self.eps_p_f)
        f = pm.utils.value(self.f_eval)
        for i in xrange(len(self.scratch3)):
            self.scratch3[i] = np.sum(eps_p_f[self.ti[i]] - f[i])

        # Compute Cholesky factor of covariance of eps_p_f, C(x,x) + V
        C_eval_value = pm.utils.value(self.C_eval)
        C_eval_shape = C_eval_value.shape
        in_chol = fc(C_eval_value, self.scratch1)
        for i in xrange(pm.utils.value(C_eval_shape)[0]):
            in_chol[i,i] += pm.utils.value(self.V) / np.alen(self.ti[i])
        info = pm.gp.linalg_utils.dpotrf_wrap(in_chol)
        if info > 0:
            raise np.linalg.LinAlgError

        # Compute covariance of f conditional on eps_p_f.
        offdiag = fc(C_eval_value, self.scratch2)
        offdiag = pm.gp.trisolve(in_chol, offdiag, uplo='U', transa='T', inplace=True)

        C_step = offdiag.T * offdiag
        C_step *= -1
        C_step += C_eval_value

        # Compute mean of f conditional on eps_p_f.
        for i in xrange(len(self.scratch3)):
            self.scratch3[i] = np.mean(eps_p_f[self.ti[i]])
        m_step = pm.utils.value(self.M_eval) + np.dot(offdiag.T, pm.gp.trisolve(in_chol,(self.scratch3 - self.M_eval.value),uplo='U',transa='T')).view(np.ndarray).ravel()

        sig_step = C_step
        info = pm.gp.linalg_utils.dpotrf_wrap(C_step.T)
        if info > 0:
            warnings.warn('Full conditional covariance was not positive definite.')
            return

        # Update value of f.
        self.f_eval.value = m_step+np.dot(sig_step,np.random.normal(size=sig_step.shape[1])).view(np.ndarray).ravel()
        # Propose the rest of the field from its conditional prior.
        self.f.rand()

class GPEvaluationAM(pm.AdaptiveMetropolis, GPEvaluationMetropolis):
    def __init__(self, submod, *args, **kwds):
        GPEvaluationMetropolis.__init__(self, submod)
        pm.AdaptiveMetropolis.__init__(self, submod.f_eval, *args, **kwds)
        
    def propose(self):
        pm.AdaptiveMetropolis.propose(self)
        self.f.rand()
    
class GPParentMetropolis(pm.Metropolis):
    pass
        
if __name__ == '__main__':
    import pylab as pl
    M = pm.gp.Mean(lambda x: np.zeros(x.shape[:-1]))
    C = pm.gp.Covariance(pm.gp.cov_funs.matern.euclidean, amp=1, scale=1, diff_degree=1)
    mesh = np.linspace(-1,1,5)
    submod = GPSubmodel('hello',M,C,mesh)
    V = .01
    epf = pm.Normal('epf', submod.f_eval, 1./V, value=np.random.normal(size=len(submod.f_eval.value)), observed=True)
    plotmesh = np.linspace(-2,2,101)
    f_eval = pm.Lambda('f_eval',lambda f=submod.f, mesh = plotmesh: f(mesh))
    MC = pm.MCMC([submod, V, epf, f_eval])        
    # MC.use_step_method(GPEvaluationGibbs, submod, V, epf)
    MC.use_step_method(GPEvaluationAM, submod)
    sm = MC.step_method_dict[submod.f_eval][0]
    
    # pl.clf()
    # for i in xrange(100):
    #     submod.f.rand()
    #     pl.plot(plotmesh, submod.f.value(plotmesh))
    # pl.plot(submod.mesh, epf.value, 'k.', markersize=10)

    MC.isample(1000,500)
    pl.clf()
    for fe in MC.trace('f_eval')[:]:
        pl.plot(plotmesh, fe)
    pl.plot(submod.mesh, epf.value, 'k.', markersize=10)

# class GPParentMetropolis(pm.Metropolis):
# 
#     """
#     M = GPParentMetropolis(stochastic, scale[, verbose, metro_method])
# 
# 
#     Wraps a pm.Metropolis instance to make it work well when one of its
#     children is a GP.
# 
# 
#     :Arguments:
# 
#         -   `stochastic`: The parent stochastic variable.
# 
#         -   `scale`: A float.
# 
#         -   `verbose`: A boolean indicating whether verbose mode should be on.
# 
#         -   `metro_method`: The pm.Metropolis subclass instance to be wrapped.
# 
# 
#     :SeeAlso: GPMetropolis, GPNormal
#     """
#     def __init__(self, stochastic=None, scale=1., verbose=False, metro_method = None, min_adaptive_scale_factor=0):
# 
#         if (stochastic is None and metro_method is None) or (stochastic is not None and metro_method is not None):
#             raise ValueError, 'Either stochastic OR metro_method should be provided, not both.'
# 
#         if stochastic is not None:
#             stochastics = [stochastic]
#             stochastic = stochastic
# 
#             # Pick the best step method to wrap if none is provided.
#             if metro_method is None:
#                 pm.StepMethodRegistry.remove(GPParentMetropolis)
#                 self.metro_method = pm.assign_method(stochastic, scale)
#                 self.metro_class = self.metro_method.__class__
#                 self.scale = scale
#                 pm.StepMethodRegistry.append(GPParentMetropolis)
# 
#         # If the user provides a step method, wrap it.
#         if metro_method is not None:
#             self.metro_method = metro_method
#             self.metro_class = metro_method.__class__
#             stochastics = metro_method.stochastics
# 
#         # Call to pm.StepMethod's init method.
#         pm.StepMethod.__init__(self, stochastics, verbose=verbose)
#         self.min_adaptive_scale_factor=min_adaptive_scale_factor
# 
#         # Extend self's children through the GP-valued stochastics
#         # and add them to the wrapped method's children.
#         fs = set([])
#         for child in self.children:
#             if isinstance(child, GP):
#                 fs.add(child)
#                 break
# 
#         self.fs = fs
#         for f in self.fs:
#             self.children |= f.extended_children
# 
#         self.metro_method.children |= self.children
#         self.verbose = verbose
# 
#         # Record all the meshes of self's GP-valued children.
#         self._id = 'GPParent_'+ self.metro_method._id
# 
#         self.C = {}
#         self.M = {}
# 
#         for f in self.fs:
# 
#             @pm.deterministic
#             def C(C=f.parents['C']):
#                 return C
# 
#             @pm.deterministic
#             def M(M=f.parents['M']):
#                 return M
# 
#             self.C[f] = C
#             self.M[f] = M
# 
# 
#         # Wrapped method's reject() method will be replaced with this.
#         def reject_with_realization(metro_method):
#             """
#             Reject the proposed values for f and stochastic.
#             """
#             if self.verbose:
#                 print self._id + ' rejecting'
#             self.metro_class.reject(metro_method)
#             if self._need_to_reject_f:
#                 for f in self.fs:
#                     f.revert()
# 
#         setattr(self.metro_method, 'reject', types.MethodType(reject_with_realization, self.metro_method, self.metro_class))
# 
# 
#         # Wrapped method's propose() method will be replaced with this.
#         def propose_with_realization(metro_method):
#             """
#             Propose a new value for f and stochastic.
#             """
# 
#             if self.verbose:
#                 print self._id + ' proposing'
#             self.metro_class.propose(metro_method)
# 
#             # FIXME: This may not be necessary anymore now that likelihoods are computed with logp_of_sum.
#             # First make sure the proposed values are OK with f's current value
#             # on its mesh, as that will not be changed here.
#             try:
#                 # FIRST make sure none of the stochastics handled by metro_method forbid their current values.
#                 for s in self.metro_method.stochastics:
#                     s.logp
#                 for f in self.fs:
#                     f.logp
#             except pm.ZeroProbability:
#                 self._need_to_reject_f = False
#                 return
# 
#             # If the jump isn't obviously bad, propose f off its mesh from its prior.
#             self._need_to_reject_f = True
#             for f in self.fs:
#                 f.random_off_mesh()
# 
#         setattr(self.metro_method, 'propose', types.MethodType(propose_with_realization, self.metro_method, self.metro_class))
# 
#     @staticmethod
#     def competence(stochastic):
#         """
#         Competence function for GPParentMetropolis
#         """
# 
#         if any([isinstance(child, GP) for child in stochastic.extended_children]):
#             return 3
#         else:
#             return 0
# 
#     # _model, accepted, rejected and _scale have to be properties that pass
#     # set-values on to the wrapped method.
#     def _get_model(self):
#         return self.model
#     def _set_model(self, model):
#         self.model = model
#         self.metro_method._model = model
#     _model = property(_get_model, _set_model)
# 
#     def _get_accepted(self):
#         return self.metro_method.accepted
#     def _set_accepted(self, number):
#         self.metro_method.accepted = number
#     accepted = property(_get_accepted, _set_accepted)
# 
#     def _get_rejected(self):
#         return self.metro_method.rejected
#     def _set_rejected(self, number):
#         self.metro_method.rejected = number
#     rejected = property(_get_rejected, _set_rejected)
# 
#     def _get_scale(self):
#         return self.metro_method.scale
#     def _set_scale(self, number):
#         self.metro_method.scale = number
#     scale = property(_get_scale, _set_scale)
# 
#     def _get_verbose(self):
#         return self._verbose
#     def _set_verbose(self, verbose):
#         self._verbose = verbose
#         self.metro_method.verbose = verbose
#     verbose = property(_get_verbose, _set_verbose)
# 
#     # Step method just passes the call on to the wrapped method.
#     # Remember that the wrapped method's reject() and propose()
#     # methods have been overridden.
#     def step(self):
#         if self.verbose:
#             print
#             print self._id + ' stepping.'
# 
#         self.metro_method.step()
# 
#     def tune(self, verbose=0):
#         if self.metro_method.adaptive_scale_factor > self.min_adaptive_scale_factor:
#             return self.metro_method.tune(verbose=verbose)
#         else:
#             if verbose > 0:
#                 print self._id + " metro_method's asf is less than min_adaptive_scale_factor, not tuning it."
#             return False
# 
# 
# class GPMetropolis(pm.Metropolis):
#     """
#     M = GPMetropolis(stochastic, scale=.01, verbose=False)
# 
#     Makes a parent of a Realization-valued stochastic take an MCMC step.
# 
#     :Arguments:
# 
#         - `stochastic`: The GP instance.
# 
#         - `scale`: A float.
# 
#         - `verbose`: A flag.
# 
#     :SeeAlso: GPParentMetropolis, GPNormal
#     """
#     def __init__(self, stochastic, scale=.1, verbose=False):
# 
#         f = stochastic
#         self.f = stochastic
#         pm.StepMethod.__init__(self, [f], verbose)
#         self._id = 'GPMetropolis_'+self.f.__name__
# 
#         self.scale = scale
#         self.verbose = verbose
# 
#         @pm.deterministic
#         def C(C=f.parents['C']):
#             return C
# 
#         @pm.deterministic
#         def M(M=f.parents['M']):
#             return M
# 
#         self.M = M
#         self.C = C
# 
#     @staticmethod
#     def competence(stochastic):
#         if isinstance(stochastic,GP):
#             return 3
#         else:
#             return 0
# 
#     def propose(self):
#         """
#         Propose a new value for f.
#         """
#         if self.verbose:
#             print self._id + ' proposing.'
# 
#         if self.f.mesh is not None:
# 
#             # Generate values for self's value on the mesh.
#             new_mesh_value = GP_array_random(M=self.f.value(self.f.mesh, regularize=False), U=self.f.C_and_U_mesh.value[0], scale=self.scale * self.adaptive_scale_factor)
# 
#             # Generate self's value using those values.
#             C = self.f.C_and_U_mesh.value[1]
#             M_obs = copy.copy(self.M.value)
#             M_obs.observe(C, self.f.mesh, new_mesh_value)
# 
#             self.f.value = Realization(M_obs, C, regularize=False)
#             self.f.value.x_sofar = self.f.mesh
#             self.f.value.f_sofar = new_mesh_value
# 
#         else:
#             self.f.value = Realization(self.M.value, self.C.value)
# 
#     def reject(self):
#         """
#         Reject proposed value for f.
#         """
#         if self.verbose:
#             print self._id + 'rejecting.'
#         # self.f.value = self.f.last_value
#         self.f.revert()
# 
#     def tune(self, *args, **kwargs):
#         return pm.StepMethod.tune(self, *args, **kwargs)
# 
#     def step(self):
#         if self.verbose:
#             print
#             print self._id + ' getting initial prior.'
#             try:
#                 clf()
#                 plot(self.f.mesh, self.f.value(self.f.mesh),'b.',markersize=8)
#             except:
#                 pass
# 
#         logp = self.f.logp
# 
# 
#         if self.verbose:
#             print self._id + ' getting initial likelihood.'
# 
#         loglike = self.loglike
#         if self.verbose:
#             try:
#                 title('logp: %i loglike: %i' %(logp, loglike))
#                 sleep(1.)
#             except:
#                 pass
# 
#         # Sample a candidate value
#         self.propose()
# 
#         if self.verbose:
#             try:
#                 plot(self.f.mesh, self.f.value(self.f.mesh),'r.',markersize=8)
#             except:
#                 pass
# 
#         # Probability and likelihood for stochastic's proposed value:
#         try:
#             logp_p = self.f.logp
#             loglike_p = self.loglike
#             if self.verbose:
#                 try:
#                     title('logp: %i loglike: %i logp_p: %i loglike_p: %i difference: %i' %(logp,loglike,logp_p,loglike_p,logp_p-logp + loglike_p - loglike))
#                     sleep(5.)
#                 except:
#                     pass
# 
#         # Reject right away if jump was to a value forbidden by self's children.
#         except pm.ZeroProbability:
# 
#             self.reject()
#             self.rejected += 1
#             if self.verbose:
#                 print self._id + ' returning.'
#                 try:
#                     title('logp: %i loglike: %i jump forbidden' %(logp, loglike))
#                 except:
#                     pass
#                 sleep(5.)
#             return
# 
#         if self.verbose:
#             print 'logp_p - logp: ', logp_p - logp
#             print 'loglike_p - loglike: ', loglike_p - loglike
# 
#         # Test
#         if np.log(np.random.random()) > logp_p + loglike_p - logp - loglike:
#             self.reject()
#             self.rejected += 1
# 
#         else:
#             self.accepted += 1
#             if self.verbose:
#                 print self._id + ' accepting'
# 
#         if self.verbose:
#             print self._id + ' returning.'
# 
# 
