# Copyright (c) Anand Patil, 2007

__docformat__ = 'reStructuredText'

__all__ = ['GP_array_random', 'GP', 'GPMetropolis', 'GPNormal', 'GPParentMetropolis']

import pymc as pm
import linalg_utils
import copy
import types
import numpy as np

from Realization import Realization
from Mean import Mean
from Covariance import Covariance
from GPutils import observe, regularize_array


def GP_array_random(M, U, scale=1.):
    b = np.dot(U.T , np.random.normal(size = U.shape[0]))
    return np.asarray(M+b*scale).squeeze()

class GP(pm.Stochastic):
    """
    f = GP('f', M, C, [mesh, doc, init_mesh_vals, mesh_eval_observed, trace, cache_depth, verbose])

    A stochastic variable valued as a Realization object.

    :Arguments:

        - M: A Mean instance or pm.deterministic variable valued as a Mean instance.

        - C: A Covariance instance or pm.deterministic variable valued as a Covariance instance.

        - mesh: The mesh on which self's log-probability will be computed. See documentation.

        - init_mesh_vals: Initial values for self over mesh.

        - mesh_eval_observed: Whether self's evaluation over mesh is known with no uncertainty.

        - trace: Whether self should be traced. See pymc documentation.

        - cache_depth: See pymc documentation.

        - verbose: An integer from 0 to 3.

    :SeeAlso: Realization, pymc.pm.Stochastic, pymc.pm.Deterministic, GPMetropolis, GPNormal
    """
    def __init__(self,
                 name,
                 M,
                 C,
                 mesh=None,
                 doc="A GP realization-valued stochastic variable",
                 init_mesh_vals = None,
                 mesh_eval_observed = False,
                 trace=True,
                 cache_depth=2,
                 verbose=False):

        self.conjugate=True

        if not isinstance(mesh, np.ndarray) and mesh is not None:
            raise ValueError, name + ": __init__ argument 'mesh' must be ndarray."

        self.mesh = regularize_array(mesh)

        # Safety
        if isinstance(M, pm.Deterministic):
            if not isinstance(M.value, Mean):
                raise ValueError,   'GP' + self.__name__ + ': Argument M must be Mean instance '+\
                                    'or pm.Deterministic valued as Mean instance, got pm.Deterministic valued as ' + M.__class__.__name__
        elif not isinstance(M, Mean):
            raise ValueError,   'GP' + self.__name__ + ': Argument M must be Mean instance '+\
                                'or pm.Deterministic valued as Mean instance, got ' + M.__class__.__name__

        if isinstance(C, pm.Deterministic):
            if not isinstance(C.value, Covariance):
                raise ValueError,   'GP' + self.__name__ + ': Argument C must be Covariance instance '+\
                                    'or pm.Deterministic valued as Covariance instance, got pm.Deterministic valued as ' + C.__class__.__name__
        elif not isinstance(C, Covariance):
            raise ValueError,   'GP' + self.__name__ + ': Argument C must be Covariance instance '+\
                                'or pm.Deterministic valued as Covariance instance, got ' + C.__class__.__name__

        # This function will be used to draw values for self conditional on self's parents.

        def random_fun(M,C):
            return Realization(M,C)
        self._random = random_fun

        if self.mesh is not None:
            self.mesh = regularize_array(self.mesh)

            @pm.deterministic(verbose=verbose-1, cache_depth=cache_depth, trace=False)
            def M_mesh(M=M, mesh=self.mesh):
                """
                The mean function evaluated on the mesh,
                cached for speed.
                """
                return M(mesh, regularize=False)
            M_mesh.__name__ = name + '_M_mesh'

            @pm.deterministic(verbose=verbose-1, cache_depth=cache_depth, trace=False)
            def C_and_U_mesh(C=C, mesh=self.mesh):
                """
                The upper-triangular Cholesky factor of the
                covariance function evaluated on the mesh,
                cached for speed.
                """
                # TODO: Will need to change this when sparse covariances
                # are available. dpotrf doesn't know about sparse covariances.

                C_old = C
                C = copy.copy(C)

                # Most covariances generate U_mesh automatically when observed,
                # and after observation they can be used to efficiently construct realizations.
                try:
                    junk1, junk2, U = C.observe(mesh, np.zeros(mesh.shape[0]), assume_full_rank=True)
                except np.linalg.LinAlgError:
                    return np.linalg.LinAlgError

                # Handle BasisCovariances separately
                if U is None:
                    try:
                        U = np.linalg.cholesky(C_old(mesh,mesh,regularize=False)).T
                    except np.linalg.LinAlgError:
                        return np.linalg.LinAlgError

                # print U.T*U - C_old(mesh,mesh,regularize=False)

                if U.shape[0] < U.shape[1]:
                    return np.linalg.LinAlgError
                return U, C

            C_and_U_mesh.__name__ = name + 'C_and_U_mesh'

            self.M_mesh = M_mesh
            self.C_and_U_mesh = C_and_U_mesh

        def logp_fun(value, M, C):
            if self.mesh is None:
                if self.verbose > 1:
                    print '\t%s: No mesh, returning 0.' % self.__name__
                return 0.
            elif self.C_and_U_mesh.value is np.linalg.LinAlgError:
                if self.verbose > 1:
                    print '\t%s: Covariance not positive definite on mesh, returning -infinity.' % self.__name__
                return -np.Inf
            else:
                if self.verbose > 1:
                    print '\t%s: Computing log-probability.' % self.__name__
                return linalg_utils.gp_array_logp(value(self.mesh), self.M_mesh.value, self.C_and_U_mesh.value[0])


        @pm.deterministic(verbose=verbose-1, cache_depth=cache_depth, trace=False)
        def init_val(M=M, C=C, init_mesh = self.mesh, init_mesh_vals = init_mesh_vals):
            if init_mesh_vals is not None:
                return Realization(M, C, init_mesh, init_mesh_vals.ravel(), regularize=False)
            else:
                return Realization(M, C)
        init_val.__name__ = name + '_init_val'

        pm.Stochastic.__init__( self,
                            logp=logp_fun,
                            doc=doc,
                            name=name,
                            parents={'M': M, 'C': C},
                            dtype=np.dtype('object'),
                            random = random_fun,
                            trace=trace,
                            value=init_val.value,
                            observed=False,
                            cache_depth=cache_depth,
                            verbose = verbose)

    def random_off_mesh(self):
        if self.mesh is not None:

            cur_mesh_value = self.value(self.mesh).copy()

            # Observe the mean on the mesh at variance 0.
            M_obs = copy.copy(self.parents.value['M'])
            C_obs = copy.copy(self.C_and_U_mesh.value[1])
            M_obs.observe(C_obs, self.mesh, cur_mesh_value)

            # Make a cheap realization.
            new_value = Realization(M_obs, C_obs, regularize=False)
            new_value.x_sofar = self.mesh
            new_value.f_sofar = cur_mesh_value

            self.value = new_value

            # self.value = Realization(self.parents.value['M'], self.parents.value['C'], init_mesh = self.mesh, init_vals = cur_mesh_value)

        else:
            self.random()

class GPParentMetropolis(pm.Metropolis):

    """
    M = GPParentMetropolis(stochastic, scale[, verbose, metro_method])


    Wraps a pm.Metropolis instance to make it work well when one of its
    children is a GP.


    :Arguments:

        -   `stochastic`: The parent stochastic variable.

        -   `scale`: A float.

        -   `verbose`: A boolean indicating whether verbose mode should be on.

        -   `metro_method`: The pm.Metropolis subclass instance to be wrapped.


    :SeeAlso: GPMetropolis, GPNormal
    """
    def __init__(self, stochastic=None, scale=1., verbose=False, metro_method = None, min_adaptive_scale_factor=0, tally=False):

        if (stochastic is None and metro_method is None) or (stochastic is not None and metro_method is not None):
            raise ValueError, 'Either stochastic OR metro_method should be provided, not both.'

        if stochastic is not None:
            stochastics = [stochastic]
            stochastic = stochastic

            # Pick the best step method to wrap if none is provided.
            if metro_method is None:
                pm.StepMethodRegistry.remove(GPParentMetropolis)
                self.metro_method = pm.assign_method(stochastic, scale)
                self.metro_class = self.metro_method.__class__
                self.scale = scale
                pm.StepMethodRegistry.append(GPParentMetropolis)

        # If the user provides a step method, wrap it.
        if metro_method is not None:
            self.metro_method = metro_method
            self.metro_class = metro_method.__class__
            stochastics = metro_method.stochastics

        # Call to pm.StepMethod's init method.
        pm.StepMethod.__init__(self, stochastics, verbose=verbose, tally=tally)
        self.min_adaptive_scale_factor=min_adaptive_scale_factor
        
        self._tuning_info = ['metro_method_adaptive_scale_factor']
        

        # Extend self's children through the GP-valued stochastics
        # and add them to the wrapped method's children.
        fs = set([])
        for child in self.children:
            if isinstance(child, GP):
                fs.add(child)
                break

        self.fs = fs
        for f in self.fs:
            self.children |= f.extended_children

        self.metro_method.children |= self.children
        self.verbose = verbose

        # Record all the meshes of self's GP-valued children.
        self._id = 'GPParent_'+ self.metro_method._id

        self.C = {}
        self.M = {}

        for f in self.fs:

            @pm.deterministic
            def C(C=f.parents['C']):
                return C

            @pm.deterministic
            def M(M=f.parents['M']):
                return M

            self.C[f] = C
            self.M[f] = M


        # Wrapped method's reject() method will be replaced with this.
        def reject_with_realization(metro_method):
            """
            Reject the proposed values for f and stochastic.
            """
            if self.verbose:
                print self._id + ' rejecting'
            self.metro_class.reject(metro_method)
            if self._need_to_reject_f:
                for f in self.fs:
                    f.revert()

        setattr(self.metro_method, 'reject', types.MethodType(reject_with_realization, self.metro_method, self.metro_class))


        # Wrapped method's propose() method will be replaced with this.
        def propose_with_realization(metro_method):
            """
            Propose a new value for f and stochastic.
            """

            if self.verbose:
                print self._id + ' proposing'
            self.metro_class.propose(metro_method)

            # First make sure the proposed values are OK with f's current value
            # on its mesh, as that will not be changed here.
            try:
                # FIRST make sure none of the stochastics handled by metro_method forbid their current values.
                for s in self.metro_method.stochastics:
                    s.logp
                for f in self.fs:
                    f.logp
            except pm.ZeroProbability:
                self._need_to_reject_f = False
                return

            # If the jump isn't obviously bad, propose f off its mesh from its prior.
            self._need_to_reject_f = True
            for f in self.fs:
                f.random_off_mesh()

        setattr(self.metro_method, 'propose', types.MethodType(propose_with_realization, self.metro_method, self.metro_class))

    @staticmethod
    def competence(stochastic):
        """
        Competence function for GPParentMetropolis
        """

        if any([isinstance(child, GP) for child in stochastic.extended_children]):
            return 3
        else:
            return 0

    # _model, accepted, rejected and _scale have to be properties that pass
    # set-values on to the wrapped method.
    def _get_model(self):
        return self.model
    def _set_model(self, model):
        self.model = model
        self.metro_method._model = model
    _model = property(_get_model, _set_model)

    def _get_accepted(self):
        return self.metro_method.accepted
    def _set_accepted(self, number):
        self.metro_method.accepted = number
    accepted = property(_get_accepted, _set_accepted)

    def _get_rejected(self):
        return self.metro_method.rejected
    def _set_rejected(self, number):
        self.metro_method.rejected = number
    rejected = property(_get_rejected, _set_rejected)

    def _get_scale(self):
        return self.metro_method.scale
    def _set_scale(self, number):
        self.metro_method.scale = number
    scale = property(_get_scale, _set_scale)

    def _get_verbose(self):
        return self._verbose
    def _set_verbose(self, verbose):
        self._verbose = verbose
        self.metro_method.verbose = verbose
    verbose = property(_get_verbose, _set_verbose)
    
    def _get_metro_method_adaptive_scale_factor(self):
        return self.metro_method.adaptive_scale_factor
    metro_method_adaptive_scale_factor = property(_get_metro_method_adaptive_scale_factor)

    # Step method just passes the call on to the wrapped method.
    # Remember that the wrapped method's reject() and propose()
    # methods have been overridden.
    def step(self):
        if self.verbose:
            print
            print self._id + ' stepping.'

        self.metro_method.step()

    def tune(self, verbose=0):
        if self.metro_method.adaptive_scale_factor > self.min_adaptive_scale_factor:
            return self.metro_method.tune(verbose=verbose)
        else:
            if verbose > 0:
                print self._id + " metro_method's asf is less than min_adaptive_scale_factor, not tuning it."
            return False


class GPMetropolis(pm.Metropolis):
    """
    M = GPMetropolis(stochastic, scale=.01, verbose=False)

    Makes a parent of a Realization-valued stochastic take an MCMC step.

    :Arguments:

        - `stochastic`: The GP instance.

        - `scale`: A float.

        - `verbose`: A flag.

    :SeeAlso: GPParentMetropolis, GPNormal
    """
    def __init__(self, stochastic, scale=.1, verbose=False, tally=False):

        f = stochastic
        self.f = stochastic
        pm.Metropolis.__init__(self, f, verbose=verbose, tally=tally)
        self._id = 'GPMetropolis_'+self.f.__name__

        self.scale = scale
        self.verbose = verbose

        @pm.deterministic
        def C(C=f.parents['C']):
            return C

        @pm.deterministic
        def M(M=f.parents['M']):
            return M

        self.M = M
        self.C = C

    @staticmethod
    def competence(stochastic):
        if isinstance(stochastic,GP):
            return 3
        else:
            return 0

    def propose(self):
        """
        Propose a new value for f.
        """
        if self.verbose:
            print self._id + ' proposing.'

        if self.f.mesh is not None:

            # Generate values for self's value on the mesh.
            new_mesh_value = GP_array_random(M=self.f.value(self.f.mesh, regularize=False), U=self.f.C_and_U_mesh.value[0], scale=self.scale * self.adaptive_scale_factor)

            # Generate self's value using those values.
            C = self.f.C_and_U_mesh.value[1]
            M_obs = copy.copy(self.M.value)
            M_obs.observe(C, self.f.mesh, new_mesh_value)

            self.f.value = Realization(M_obs, C, regularize=False)
            self.f.value.x_sofar = self.f.mesh
            self.f.value.f_sofar = new_mesh_value

        else:
            self.f.value = Realization(self.M.value, self.C.value)

    def reject(self):
        """
        Reject proposed value for f.
        """
        if self.verbose:
            print self._id + 'rejecting.'
        # self.f.value = self.f.last_value
        self.f.revert()

    def tune(self, *args, **kwargs):
        return pm.StepMethod.tune(self, *args, **kwargs)

    def step(self):
        if self.verbose:
            print
            print self._id + ' getting initial prior.'
            try:
                clf()
                plot(self.f.mesh, self.f.value(self.f.mesh),'b.',markersize=8)
            except:
                pass

        logp = self.f.logp


        if self.verbose:
            print self._id + ' getting initial likelihood.'

        loglike = self.loglike
        if self.verbose:
            try:
                title('logp: %i loglike: %i' %(logp, loglike))
                sleep(1.)
            except:
                pass

        # Sample a candidate value
        self.propose()

        if self.verbose:
            try:
                plot(self.f.mesh, self.f.value(self.f.mesh),'r.',markersize=8)
            except:
                pass

        # Probability and likelihood for stochastic's proposed value:
        try:
            logp_p = self.f.logp
            loglike_p = self.loglike
            if self.verbose:
                try:
                    title('logp: %i loglike: %i logp_p: %i loglike_p: %i difference: %i' %(logp,loglike,logp_p,loglike_p,logp_p-logp + loglike_p - loglike))
                    sleep(5.)
                except:
                    pass

        # Reject right away if jump was to a value forbidden by self's children.
        except pm.ZeroProbability:

            self.reject()
            self.rejected += 1
            if self.verbose:
                print self._id + ' returning.'
                try:
                    title('logp: %i loglike: %i jump forbidden' %(logp, loglike))
                except:
                    pass
                sleep(5.)
            return

        if self.verbose:
            print 'logp_p - logp: ', logp_p - logp
            print 'loglike_p - loglike: ', loglike_p - loglike

        # Test
        if np.log(np.random.random()) > logp_p + loglike_p - logp - loglike:
            self.reject()
            self.rejected += 1

        else:
            self.accepted += 1
            if self.verbose:
                print self._id + ' accepting'

        if self.verbose:
            print self._id + ' returning.'



class GPNormal(pm.Gibbs):
    """
    S = GPNormal(f, obs_mesh, obs_V, obs_vals)


    Causes GP f and its mesh_eval attribute
    to take a pm.Gibbs step. Applies to f in the following submodel:


    obs_vals ~ N(f(obs_mesh)+offset, obs_V)
    f ~ GP(M,C)


    :SeeAlso: GPMetropolis, GPParentMetropolis
    """

    def __init__(self, f, obs_mesh, obs_V, obs_vals, offset=None, same_mesh=False):

        if not isinstance(f, GP):
            raise ValueError, 'GPNormal can only handle GPs, cannot handle '+f.__name__

        pm.StepMethod.__init__(self, variables=[f])
        self.f = f
        self._id = 'GPNormal_'+self.f.__name__
        self.same_mesh = same_mesh
        if self.same_mesh:
            obs_mesh = self.f.mesh

        @pm.deterministic
        def obs_mesh(obs_mesh=obs_mesh):
            return regularize_array(obs_mesh)

        @pm.deterministic
        def obs_V(obs_V=obs_V, obs_mesh = obs_mesh):
            return np.resize(obs_V, obs_mesh.shape[0])

        @pm.deterministic
        def obs_vals(obs_vals=obs_vals):
            return obs_vals

        # M_local and C_local's values are copies of f's M and C parents,
        # observed according to obs_mesh and obs_V.
        @pm.deterministic
        def C_local(C = f.parents['C'], obs_mesh = obs_mesh, obs_V = obs_V):
            """
            The covariance, observed according to the children,
            with supporting information.
            """
            C_local = copy.copy(C)
            relevant_slice, obs_vals_new, junk = C_local.observe(obs_mesh, obs_V)
            return (C_local, relevant_slice, obs_vals_new)

        @pm.deterministic
        def M_local(C_local = C_local, obs_vals = obs_vals, M = f.parents['M'], offset=offset):
            """
            The mean function, observed according to the children.
            """
            relevant_slice = C_local[1]
            obs_mesh_new = C_local[2]

            obs_vals = obs_vals.ravel()[relevant_slice]
            if offset is not None:
                obs_vals = obs_vals - offset.ravel()[relevant_slice]

            M_local = copy.copy(M)
            M_local.observe(C_local[0], obs_mesh_new, obs_vals)
            return M_local

        if self.same_mesh:
            @pm.deterministic
            def U_mesh(C=C_local, mesh=self.f.mesh):
                return C[0].cholesky(mesh)

            @pm.deterministic
            def M_mesh(M=M_local, mesh=self.f.mesh):
                return M(mesh)

            self.U_mesh = U_mesh
            self.M_mesh = M_mesh

        self.obs_mesh = obs_mesh
        self.obs_vals = obs_vals
        self.obs_V = obs_V
        self.M_local = M_local
        self.C_local = C_local

    def step(self):
        """
        Samples self.f's value from its conditional distribution.
        """
        try:
            if not self.same_mesh:
                self.f.value = Realization(self.M_local.value, self.C_local.value[0])
            else:
                # Generate values for self's value on the mesh.
                new_mesh_value = GP_array_random(M=self.M_mesh.value, U=self.U_mesh.value)

                # Generate self's value using those values.
                C = self.f.C_and_U_mesh.value[1]
                M_obs = copy.copy(self.f.parents.value['M'])
                M_obs.observe(C, self.f.mesh, new_mesh_value)

                self.f.value = Realization(M_obs, C, regularize=False)
                self.f.value.x_sofar = self.f.mesh
                self.f.value.f_sofar = new_mesh_value

        except np.linalg.LinAlgError:
            print 'Covariance was numerically singular when stepping f, trying again.'
            self.step()
