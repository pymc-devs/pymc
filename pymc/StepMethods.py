from __future__ import division

import numpy as np
from .utils import msqrt, check_type, round_array, float_dtypes, integer_dtypes, bool_dtypes, safe_len, find_generations, logp_of_set, symmetrize, logp_gradient_of_set
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs, exp, where, isscalar, iterable, multiply, transpose, tri, pi
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import pinv, cholesky
from numpy.random import randint, random
from numpy.random import normal as rnormal
from numpy.random import poisson as rpoisson
from .PyMCObjects import Stochastic, Potential, Deterministic
from .Container import Container
from .Node import ZeroProbability, Node, Variable, StochasticBase
from pymc.decorators import prop
from . import distributions
from copy import copy
from .InstantiationDecorators import deterministic
import pdb, warnings, sys
import inspect

from . import six
from .six import print_

__docformat__='reStructuredText'


# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the step method, and a method to return that state in a dict. Added an id.
# TODO: Test cases for binary and discrete Metropolises.

conjugate_Gibbs_competence = 0
nonconjugate_Gibbs_competence = 0

class AdaptationError(ValueError): pass


__all__=['DiscreteMetropolis', 'Metropolis', 'PDMatrixMetropolis', 'StepMethod', 'assign_method',  'pick_best_methods', 'StepMethodRegistry', 'NoStepper', 'BinaryMetropolis', 'AdaptiveMetropolis','Gibbs','conjugate_Gibbs_competence', 'nonconjugate_Gibbs_competence', 'DrawFromPrior']


StepMethodRegistry = []

def pick_best_methods(stochastic):
    """
    Picks the StepMethods best suited to handle
    a stochastic variable.
    """

    # Keep track of most competent methohd
    max_competence = 0
    # Empty set of appropriate StepMethods
    best_candidates = set([])

    # Loop over StepMethodRegistry
    for method in StepMethodRegistry:

        # Parse method and its associated competence
        try:
            competence = method.competence(stochastic)
        except:
#             print_('\n\tWarning, there was an error while step method %s assessed its competence \n \
# \tto handle stochastic %s. It is being excluded from consideration.\n' \
#                     %(method.__name__, stochastic))
            competence = 0

        # If better than current best method, promote it
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence

        # If same competence, add it to the set of best methods
        elif competence == max_competence:
            best_candidates.add(method)

    if max_competence<=0:
        raise ValueError('Maximum competence reported for stochastic %s is <= 0... you may need to write a custom step method class.' % stochastic.__name__)

    # print_(s.__name__ + ': ', best_candidates, ' ', max_competence)
    return best_candidates

def assign_method(stochastic, scale=None, verbose=-1):
    """
    Returns a step method instance to handle a
    variable. If several methods have the same competence,
    it picks one arbitrarily (using set.pop()).
    """

    # Retrieve set of best candidates
    best_candidates = pick_best_methods(stochastic)

    # Randomly grab and appropriate method
    method = best_candidates.pop()

    failure_header = """Failed attempting to automatically assign step method class %s
to stochastic variable %s. Try setting %s's competence method to return 0
and manually assigning it when appropriate. See the user guide.

Error message: """%(method.__name__, stochastic.__name__, method.__name__)

    try:
        if scale:
            out = method(stochastic, scale = scale, verbose=verbose)
        else:
            out = method(stochastic, verbose=verbose)
    except:
        a,b,c = sys.exc_info()
        try:
            args = list(b.args)
        except AttributeError:
            args = []
        args.append(failure_header)
        b.args = args
        six.reraise(a, b, c)
    return out


class StepMethodMeta(type):
    """
    Automatically registers new step methods if they can be automatically assigned:
    if their init method has one and only one required argument.
    """
    def __init__(cls, name, bases, dict):
        type.__init__(cls, name, bases, dict)
        args, varargs, varkw, defaults = inspect.getargspec(cls.__init__)
        auto_assignment_OK = False
        if len(args) == 2:
            auto_assignment_OK = True
        elif len(args)>2:
            if defaults is not None:
                if len(defaults) == len(args)-2:
                    auto_assignment_OK = True
        elif len(args) == 1 and varargs is not None:
            auto_assignment_OK = True

        if auto_assignment_OK:
            StepMethodRegistry.append(cls)


class StepMethod(object):
    """
    This object knows how to make Stochastics take single MCMC steps.
    Its step() method will be called by Model at every MCMC iteration.

    :Parameters:
          -variables : list, array or set
            Collection of PyMCObjects

          - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high. Setting to -1 (Default) allows verbosity to be set by sampler.

    Externally-accessible attributes:
      stochastics:   The Stochastics over which self has jurisdiction which have observed = False.
      children: The combined children of all Variables over which self has jurisdiction.
      parents:  The combined parents of all Nodes over which self has jurisdiction, as a set.
      loglike:  The summed log-probability of self's children conditional on all of self's
                  Variables' current values. These will be recomputed only as necessary.
                  This descriptor should eventually be written in C.

    Externally accesible methods:
      sample(): A single MCMC step for all the Stochastics over which self has
        jurisdiction. Must be overridden in subclasses.
      tune(): Tunes proposal distribution widths for all self's Stochastics.
      competence(s): Examines Stochastic instance s and returns self's
        competence to handle it, on a scale of 0 to 3.

    To instantiate a StepMethod called S with jurisdiction over a
    sequence/set N of Nodes:

      >>> S = StepMethod(N)

    :SeeAlso: Metropolis, Sampler.
    """

    def __init__(self, variables, verbose=-1, tally=False):
        # StepMethod initialization

        if not iterable(variables) or isinstance(variables, Node):
            variables = [variables]

        self.stochastics = set()
        self.children = set()
        self.parents = set()
        self.tally = tally

        self._state = []
        self._tuning_info = []
        self.verbose = verbose

        # File away the variables
        for variable in variables:
            # Sort.

            if isinstance(variable,Stochastic) and not variable.observed:
                self.stochastics.add(variable)

        if len(self.stochastics)==0:
            raise ValueError('No stochastics provided.')

        # Find children, no need to find parents; each variable takes care of those.
        for variable in variables:
            self.children |= variable.children
            for parent in six.itervalues(variable.parents):
                if isinstance(parent, Variable):
                    self.parents.add(parent)

        self.children = set([])
        self.parents = set([])
        for s in self.stochastics:
            self.children |= s.extended_children
            self.parents |= s.extended_parents

        # Remove own stochastics from children and parents.
        self.children -= self.stochastics
        self.parents -= self.stochastics

        # self.markov_blanket is a list, because we want self.stochastics to have the chance to
        # raise ZeroProbability exceptions before self.children.
        self.markov_blanket = list(self.stochastics)+list(self.children)

        # ID string for verbose feedback
        self._id = self.__class__.__name__ + '_' + '_'.join([s.__name__ for s in self.stochastics])

    def step(self):
        """
        Specifies single step of step method.
        Must be overridden in subclasses.
        """
        pass

    @staticmethod
    def competence(s):
        """
        This function is used by Sampler to determine which step method class
        should be used to handle stochastic variables.

        Return value should be a competence
        score from 0 to 3, assigned as follows:

        0:  I can't handle that variable.
        1:  I can handle that variable, but I'm a generalist and
            probably shouldn't be your top choice (Metropolis
            and friends fall into this category).
        2:  I'm designed for this type of situation, but I could be
            more specialized.
        3:  I was made for this situation, let me handle the variable.

        In order to be eligible for inclusion in the registry, a sampling
        method's init method must work with just a single argument, a
        Stochastic object.

        If you want to exclude a particular step method from
        consideration for handling a variable, do this:

        Competence functions MUST be called 'competence' and be decorated by the
        '@staticmethod' decorator. Example:

            @staticmethod
            def competence(s):
                if isinstance(s, MyStochasticSubclass):
                    return 2
                else:
                    return 0

        :SeeAlso: pick_best_methods, assign_method
        """
        return 0


    def tune(self, *args, **kwargs):
        return False

    def _get_loglike(self):
        # Fetch log-probability (as sum of childrens' log probability)
        sum = logp_of_set(self.children)
        if self.verbose>2:
            print_('\t' + self._id + ' Current log-likelihood ', sum)
        return sum

    # Make get property for retrieving log-probability
    loglike = property(fget = _get_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, with self.stochastics removed.")

    def _get_logp_plus_loglike(self):
        sum = logp_of_set(self.markov_blanket)
        if self.verbose>2:
            print_('\t' + self._id + ' Current log-likelihood plus current log-probability', sum)
        return sum

    # Make get property for retrieving log-probability
    logp_plus_loglike = property(fget = _get_logp_plus_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, and self.stochastics.")

    def _get_logp_gradient(self):
        return logp_gradient_of_set(self.stochastics, self.markov_blanket)
    
    logp_gradient = property(fget = _get_logp_gradient)
    
    def current_state(self):
        """Return a dictionary with the current value of the variables defining
        the state of the step method."""
        state = {}
        for s in self._state:
            state[s] = getattr(self, s)
        return state


    @prop
    def ratio():
        """Acceptance ratio"""
        def fget(self):
            return self.accepted/(self.accepted + self.rejected)
        return locals()

StepMethod = six.with_metaclass(StepMethodMeta, StepMethod)

class NoStepper(StepMethod):
    """
    Step and tune methods do nothing.

    Useful for holding stochastics constant without setting observed=True.
    """
    def step(self):
        pass
    def tune(self, *args, **kwargs):
        pass

# The default StepMethod, which Model uses to handle singleton stochastics.
class Metropolis(StepMethod):
    """
    The default StepMethod, which Model uses to handle singleton, continuous variables.

    Applies the one-at-a-time Metropolis-Hastings algorithm to the Stochastic over which self has jurisdiction.

    To instantiate a Metropolis called M with jurisdiction over a Stochastic P:

      >>> M = Metropolis(P, scale=1, proposal_sd=None, dist=None)

    :Arguments:
    - s : Stochastic
            The variable over which self has jurisdiction.

    - scale (optional) : number
            The proposal jump width is set to scale * variable.value.

    - proposal_sd (optional) : number or vector
            The proposal jump width is set to proposal_sd.

    - proposal_distribution (optional) : string
            The proposal distribution. May be 'Normal',
            'Prior' or None. If None is provided, a proposal distribution is chosen
            by examining P.value's type.

    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high. Setting to -1 (default) allows verbosity to be turned on by sampler.

    :SeeAlso: StepMethod, Sampler.
    """

    def __init__(self, stochastic, scale=1., proposal_sd=None, proposal_distribution=None, verbose=-1, tally=True, check_before_accepting=True):
        # Metropolis class initialization

        # Initialize superclass
        StepMethod.__init__(self, [stochastic], tally=tally)

        # Initialize hidden attributes
        self.proposal_sd = proposal_sd

        self.adaptive_scale_factor = 1.
        self.accepted = 0.
        self.rejected = 0.
        self._state = ['rejected', 'accepted', 'adaptive_scale_factor', 'proposal_sd', 'proposal_distribution', 'check_before_accepting']
        self._tuning_info = ['adaptive_scale_factor']
        self.check_before_accepting = check_before_accepting
        self.proposal_sd=proposal_sd

        # Set public attributes
        self.stochastic = stochastic
        if verbose > -1:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose

        if proposal_distribution != "Prior":
            # Avoid zeros when setting proposal variance
            if proposal_sd is None:
                if all(self.stochastic.value != 0.):
                    self.proposal_sd = ones(shape(self.stochastic.value)) * abs(self.stochastic.value) * scale
                else:
                    self.proposal_sd = ones(shape(self.stochastic.value)) * scale

            # Initialize proposal deviate with array of zeros
            self.proposal_deviate = zeros(shape(self.stochastic.value), dtype=float)

            # Determine size of stochastic
            if isinstance(self.stochastic.value, ndarray):
                self._len = len(self.stochastic.value.ravel())
            else:
                self._len = 1

        #else: self.proposal_sd = None # Probably unnecessary
        # If no dist argument is provided, assign a proposal distribution automatically.
        if not proposal_distribution:

            # Pick Gaussian by default
            self.proposal_distribution = "Normal"

        else:
            
            if proposal_distribution.capitalize() in self._valid_proposals:
                self.proposal_distribution = proposal_distribution
            else: 
                raise ValueError("Invalid proposal distribution '%s' specified for Metropolis sampler." % proposal_distribution)
    
    _valid_proposals = ['Normal', 'Prior']

    @staticmethod
    def competence(s):
        """
        The competence function for Metropolis
        """
        if s.dtype is None:
            return .5

        if not s.dtype in float_dtypes:
            # If the stochastic's binary or discrete, I can't do it.
            return 0
        else:
            return 1

    def hastings_factor(self):
        """
        If this is a Metropolis-Hastings method (proposal is not symmetric random walk),
        this method should return log(back_proposal) - log(forward_proposal).
        """
        return 0.

    def step(self):
        """
        The default step method applies if the variable is floating-point
        valued, and is not being proposed from its prior.
        """

        # Probability and likelihood for s's current value:

        if self.verbose>2:
            print_()
            print_(self._id + ' getting initial logp.')

        if self.proposal_distribution == "Prior":
            logp = self.loglike
        else:
            logp = self.logp_plus_loglike

        if self.verbose>2:
            print_(self._id + ' proposing.')
            
        # Sample a candidate value
        self.propose()

        # Probability and likelihood for s's proposed value:
        try:
            if self.proposal_distribution == "Prior":
                logp_p = self.loglike
                # Check for weirdness before accepting jump
                if self.check_before_accepting:
                    self.stochastic.logp
            else:
                logp_p = self.logp_plus_loglike

        except ZeroProbability:

            # Reject proposal
            if self.verbose>2:
                print_(self._id + ' rejecting due to ZeroProbability.')
            self.reject()

            # Increment rejected count
            self.rejected += 1

            if self.verbose>2:
                print_(self._id + ' returning.')
            return

        if self.verbose>2:
            print_('logp_p - logp: ', logp_p - logp)

        HF = self.hastings_factor()

        # Evaluate acceptance ratio
        if log(random()) > logp_p - logp + HF:

            # Revert s if fail
            self.reject()

            # Increment rejected count
            self.rejected += 1
            if self.verbose > 2:
                print_(self._id + ' rejecting')
        else:
            # Increment accepted count
            self.accepted += 1
            if self.verbose > 2:
                print_(self._id + ' accepting')

        if self.verbose > 2:
            print_(self._id + ' returning.')

    def tune(self, *args, **kwargs):
        if self.proposal_distribution == "Prior":
            return False
        else:
            return StepMethod.tune(self, *args, **kwargs)

    def reject(self):
        # Sets current s value to the last accepted value
        # self.stochastic.value = self.stochastic.last_value
        self.stochastic.revert()

    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self.proposal_distribution is "Normal" (i.e. no proposal specified).
        """
        if self.proposal_distribution == "Normal":
            self.stochastic.value = rnormal(self.stochastic.value, self.adaptive_scale_factor * self.proposal_sd, size=self.stochastic.value.shape)
        elif self.proposal_distribution == "Prior":
            self.stochastic.random()

    def tune(self, divergence_threshold=1e10, verbose=0):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate of the last k proposals:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10

        This method is called exclusively during the burn-in period of the
        sampling algorithm.

        May be overridden in subclasses.
        """

        if self.verbose > -1:
            verbose = self.verbose

        # Verbose feedback
        if verbose > 0:
            print_('\t%s tuning:' % self._id)

        # Flag for tuning state
        tuning = True

        # Calculate recent acceptance rate
        if not (self.accepted + self.rejected): return tuning
        acc_rate = self.accepted / (self.accepted + self.rejected)


        # Switch statement
        if acc_rate<0.001:
            # reduce by 90 percent
            self.adaptive_scale_factor *= 0.1
        elif acc_rate<0.05:
            # reduce by 50 percent
            self.adaptive_scale_factor *= 0.5
        elif acc_rate<0.2:
            # reduce by ten percent
            self.adaptive_scale_factor *= 0.9
        elif acc_rate>0.95:
            # increase by factor of ten
            self.adaptive_scale_factor *= 10.0
        elif acc_rate>0.75:
            # increase by double
            self.adaptive_scale_factor *= 2.0
        elif acc_rate>0.5:
            # increase by ten percent
            self.adaptive_scale_factor *= 1.1
        else:
            tuning = False

        # Re-initialize rejection count
        self.rejected = 0.
        self.accepted = 0.

        # More verbose feedback, if requested
        if verbose > 0:
            if hasattr(self, 'stochastic'):
                print_('\t\tvalue:', self.stochastic.value)
            print_('\t\tacceptance rate:', acc_rate)
            print_('\t\tadaptive scale factor:', self.adaptive_scale_factor)
            print_()

        return tuning

class PDMatrixMetropolis(Metropolis):
    """Metropolis sampler with proposals customised for symmetric positive definite matrices"""
    def __init__(self, stochastic, scale=1., proposal_sd=None, verbose=-1, tally=True):
        Metropolis.__init__(self, stochastic, scale=scale, proposal_sd=proposal_sd, proposal_distribution="Normal", verbose=verbose, tally=tally)

    @staticmethod
    def competence(s):
        """
        The competence function for MatrixMetropolis
        """
        # MatrixMetropolis handles the Wishart family, which are valued as
        # _symmetric_ matrices.
        if any([isinstance(s,cls) for cls in [distributions.Wishart,distributions.InverseWishart,distributions.WishartCov]]):
            return 2
        else:
            return 0

    def propose(self):
        """
        Proposals for positive definite matrix using random walk deviations on the Cholesky
        factor of the current value.
        """

        # Locally store size of matrix
        dims = self.stochastic.value.shape

        # Add normal deviate to value and symmetrize
        dev =  rnormal(0, self.adaptive_scale_factor * self.proposal_sd, size=dims)
        symmetrize(dev)

        # Replace
        self.stochastic.value = dev + self.stochastic.value


class Gibbs(Metropolis):
    """
    Base class for the Gibbs step methods
    """
    def __init__(self, stochastic, verbose=-1):
        Metropolis.__init__(self, stochastic, verbose=verbose, tally=False)

    # Override Metropolis's competence.
    competence = classmethod(StepMethod.competence)

    def step(self):
        if not self.conjugate:
            logp = self.stochastic.logp

        self.propose()

        if not self.conjugate:

            try:
                logp_p = self.stochastic.logp
            except ZeroProbability:
                self.reject()

            if log(random()) > logp_p - logp:
                self.reject()

    def tune(self, *args, **kwargs):
        return False

    def propose(self):
        raise NotImplementedError('The Gibbs class has to be subclassed, it is not usable directly.')


class DrawFromPrior(StepMethod):
    """
    Handles dataless submodels.
    """
    def __init__(self, variables, generations, verbose=-1):
        StepMethod.__init__(self, variables, verbose, tally=False)
        self.generations = generations

        # Some variables (eg GP) may not have logp attributes, so don't try to
        # evaluate their logps.
        self.variables_with_logp = set([])
        for s in self.markov_blanket:
            try:
                s.logp
                self.variables_with_logp.add(s)
            except:
                pass
    
    def get_logp_plus_loglike(self):
        return logp_of_set(self.variables_with_logp)
    logp_plus_loglike = property(get_logp_plus_loglike)

    def step(self):
        jumped = []
        try:
            for generation in self.generations:
                for s in generation:
                    s.rand()
                    jumped.append(s)
            self.logp_plus_loglike
        except ZeroProbability:
            if self.verbose > 2:
                forbidden = []
                for generation in self.generations:
                    for s in self.stochastics:
                        try:
                            s.logp
                        except ZeroProbability:
                            forbidden.append(s.__name__)
                print_('DrawFromPrior jumped stochastics %s to value forbidden by objects %s, rejecting.'%(', '.join(s.__name__ for s in jumped),', '.join(forbidden)))
            warnings.warn('DrawFromPrior jumped to forbidden value')
            for s in jumped:
                s.revert()

    @classmethod
    def competence(s):
        # Dataless gets assigned specially before other step methods.
        return 0


class NoStepper(StepMethod):
    """
    Step and tune methods do nothing.

    Useful for holding stochastics constant without setting observed=True.
    """
    def step(self, *args, **kwargs):
        pass
    def tune(self, *args, **kwargs):
        return False

class DiscreteMetropolis(Metropolis):
    """
    Just like Metropolis, but rounds the variable's value.
    Good for discrete stochastics.
    """
    def __init__(self, stochastic, scale=1., proposal_sd=None, proposal_distribution="Poisson", positive=False, verbose=-1, tally=True):
        # DiscreteMetropolis class initialization

        # Initialize superclass
        Metropolis.__init__(self, stochastic, scale=scale, proposal_sd=proposal_sd, proposal_distribution=proposal_distribution, verbose=verbose, tally=tally)

        # Flag for positive-only values
        self._positive = positive
        
    _valid_proposals = ['Poisson', 'Normal', 'Prior']

    @staticmethod
    def competence(stochastic):
        """
        The competence function for DiscreteMetropolis.
        """
        if stochastic.dtype in integer_dtypes:
            return 1
        else:
            return 0


    def propose(self):
        # Propose new values using normal distribution

        if self.proposal_distribution == "Normal":

            # New normal deviate, centred on current value
            new_val = rnormal(self.stochastic.value, self.adaptive_scale_factor * self.proposal_sd)

            # Round before setting proposed value
            self.stochastic.value = round_array(new_val)

        elif self.proposal_distribution == "Poisson":

            k = shape(self.stochastic.value)
            # Add or subtract (equal probability) Poisson sample
            new_val = self.stochastic.value + rpoisson(self.adaptive_scale_factor * self.proposal_sd) * (-ones(k))**(random(k)>0.5)

            if self._positive:
                # Enforce positive values
                self.stochastic.value = abs(new_val)
            else:
                self.stochastic.value = new_val

        elif self.proposal_distribution == "Prior":
            self.stochastic.random()

# TODO Implement independence sampler for BinaryMetropolis

class BinaryMetropolis(Metropolis):
    """
    Like Metropolis, but with a modified step() method.
    Good for binary variables.

    """

    def __init__(self, stochastic, p_jump=.1, proposal_distribution=None, verbose=-1, tally=True):
        # BinaryMetropolis class initialization

        # Initialize superclass
        Metropolis.__init__(self, stochastic, proposal_distribution=proposal_distribution, verbose=verbose, tally=tally)

        self._state.remove('proposal_sd')

        # adaptive_scale_factor controls the jump probability
        self.adaptive_scale_factor = log(1.-p_jump) / log(.5)

    @staticmethod
    def competence(stochastic):
        """
        The competence function for Binary One-At-A-Time Metropolis
        """
        if stochastic.dtype in bool_dtypes:
            return 2
        
        elif type(stochastic) is distributions.Bernoulli:
            return 2
            
        else:
            return 0

    def step(self):
        if not isscalar(self.stochastic.value):
            Metropolis.step(self)
        else:

            # See what log-probability of True is.
            self.stochastic.value = True

            try:
                logp_true = self.logp_plus_loglike
            except ZeroProbability:
                self.stochastic.value = False
                return

            # See what log-probability of False is.
            self.stochastic.value = False

            try:
                logp_false = self.logp_plus_loglike
            except ZeroProbability:
                self.stochastic.value = True
                return

            # Test
            p_true = exp(logp_true)
            p_false = exp(logp_false)

            if self.verbose>2:
                print_("""%s step information:
    - logp_true: %f
    - logp_false: %f
    - p_true: %f
    - p_false: %f
                """ % (self._id, logp_true, logp_false, p_true, p_false))

            # Stochastically set value according to relative
            # probabilities of True and False
            if random() > p_false / (p_true + p_false):
                if self.verbose > 2:
                    print_("%s setting %s's value to True." % (self._id, self.stochastic))
                self.stochastic.value = True
            elif self.verbose > 2:
                print_("%s setting %s's value to False." % (self._id, self.stochastic))


    def propose(self):
        # Propose new values

        if self.proposal_distribution == 'Prior':
            self.stochastic.random()
        else:
            # Convert adaptive_scale_factor to a jump probability
            p_jump = 1.-.5**self.adaptive_scale_factor

            rand_array = random(size=shape(self.stochastic.value))
            new_value = copy(self.stochastic.value)
            # Locations where switches occur, according to p_jump
            switch_locs = where(rand_array<p_jump)
            if shape(new_value):
                new_value[switch_locs] = True - new_value[switch_locs]
            else:
                new_value = True - new_value
            self.stochastic.value = new_value


class AdaptiveMetropolis(StepMethod):
    """
    The AdaptativeMetropolis (AM) sampling algorithm works like a regular
    Metropolis, with the exception that stochastic parameters are block-updated
    using a multivariate jump distribution whose covariance is tuned during
    sampling. Although the chain is non-Markovian, i.e. the proposal
    distribution is asymmetric, it has correct ergodic properties. See
    (Haario et al., 2001) for details.

    :Parameters:
      - stochastic : PyMC objects
          Stochastic objects to be handled by the AM algorith,

      - cov : array
          Initial guess for the covariance matrix C. If it is None, the 
          covariance will be estimated using the scales dictionary if provided, 
          the existing trace if available, or the current stochastics value. 
          It is suggested to provide a sensible guess for the covariance, and 
          not rely on the automatic assignment from stochastics value. 

      - delay : int
          Number of steps before the empirical covariance is computed. If greedy
          is True, the algorithm waits for delay *accepted* steps before computing
          the covariance.

      - interval : int
          Interval between covariance updates. Higher dimensional spaces require 
          more samples to obtain reliable estimates for the covariance updates. 

      - greedy : bool
          If True, only the accepted jumps are tallied in the internal trace
          until delay is reached. This is useful to make sure that the empirical
          covariance has a sensible structure.

      - shrink_if_necessary : bool
          If True, the acceptance rate is checked when the step method tunes. If
          the acceptance rate is small, the proposal covariance is shrunk according
          to the following rule:

          if acc_rate < .001:
              self.C *= .01
          elif acc_rate < .01:
              self.C *= .25
              
      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is
          guessed from the trace (it if exists) or the objects value, alt
          
      - verbose : int
          Controls the verbosity level.


    :Notes:
    Use the methods: `cov_from_scales`, `cov_from_trace` and `cov_from_values` for
    more control on the creation of an initial covariance matrix. A lot of problems
    can be avoided with a good initial covariance and long enough intervals between
    covariance updates. That is, do not compensate for a bad covariance guess by 
    reducing the interval between updates thinking the covariance matrix will
    converge more rapidly. 
    

    :Reference:
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
    """
    def __init__(self, stochastic, cov=None, delay=1000, interval=200, greedy=True, shrink_if_necessary=False, scales=None, verbose=-1, tally=False):

        # Verbosity flag
        self.verbose = verbose

        self.accepted = 0
        self.rejected = 0

        if not np.iterable(stochastic) or isinstance(stochastic, Variable):
            stochastic = [stochastic]

        # Initialize superclass
        StepMethod.__init__(self, stochastic, verbose, tally)

        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += ['accepted', 'rejected', '_trace_count', '_current_iter', 'C', 'proposal_sd',
        '_proposal_deviate', '_trace', 'shrink_if_necessary']
        self._tuning_info = ['C']

        self.proposal_sd = None
        self.shrink_if_necessary=shrink_if_necessary

        # Number of successful steps before the empirical covariance is computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy

        # Initialization methods
        self.check_type()
        self.dimension()
        
        # Set the initial covariance using cov, or the following fallback mechanisms:
        # 1. If scales is provided, use it. 
        # 2. If a trace is present, compute the covariance matrix empirically from it. 
        # 3. Use the stochastics value as a guess of the variance. 
        if cov is not None:
            self.C = cov
        elif scales:
            self.C = self.cov_from_scales(scales)
        else:
            try:
                self.C = self.cov_from_trace()
            except AttributeError:
                self.C = self.cov_from_value(100.)
    
        self.updateproposal_sd()

        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy
        # sampling can be done during warm-up period.
        self._trace_count = 0
        self._current_iter = 0

        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []

        if self.verbose >= 2:
            print_("Initialization...")
            print_('Dimension: ', self.dim)
            print_("C_0: ", self.C)
            print_("Sigma: ", self.proposal_sd)


    @staticmethod
    def competence(stochastic):
        """
        The competence function for AdaptiveMetropolis.
        The AM algorithm is well suited to deal with multivariate
        parameters.
        """
        # if not stochastic.dtype in float_dtypes and not stochastic.dtype in integer_dtypes:
        #             return 0
        #             # Algorithm is not well-suited to sparse datasets. Dont use if less than
        #             # 25 percent of values are nonzero
        #         if not getattr(stochastic, 'mask', None) is None:
        #             return 0
        #         if np.alen(stochastic.value) == 1:
        #             return 0
        #         elif np.alen(stochastic.value) < 5:
        #             return 2
        #         elif (len(stochastic.value.nonzero()[0]) > 0.25*len(stochastic.value)):
        #             return 2
        #         else:
        #             return 0
        return 0
                
    def cov_from_value(self, scaling):
        """Return a covariance matrix for the jump distribution using 
        the actual value of the stochastic as a guess of their variance, 
        divided by the `scaling` argument. 
        
        Note that this is likely to return a poor guess. 
        """
        rv = []
        for s in self.stochastics:
            rv.extend(np.ravel(s.value).copy())
        
        # Remove 0 values since this would lead to quite small jumps... 
        arv = np.array(rv)
        arv[arv==0] = 1.

        # Create a diagonal covariance matrix using the scaling factor.
        return np.eye(self.dim)*np.abs(arv)/scaling


    def cov_from_scales(self, scales):
        """Return a covariance matrix built from a dictionary of scales.
        
        `scales` is a dictionary keyed by stochastic instances, and the 
        values refer are the variance of the jump distribution for each 
        stochastic. If a stochastic is a sequence, the variance must
        have the same length. 
        """
       
        # Get array of scales
        ord_sc = []
        for stochastic in self.stochastics:
            ord_sc.append(np.ravel(scales[stochastic]))
        ord_sc = np.concatenate(ord_sc)

        if np.squeeze(ord_sc).shape[0] != self.dim:
            raise ValueError("Improper initial scales, dimension don't match", \
                (np.squeeze(ord_sc), self.dim))
        
        # Scale identity matrix
        return np.eye(self.dim)*ord_sc

    def cov_from_trace(self, trace=slice(None)):
        """Define the jump distribution covariance matrix from the object's 
        stored trace.
        
        :Parameters:
        - `trace` : slice or int
          A slice for the stochastic object's trace in the last chain, or a 
          an integer indicating the how many of the last samples will be used.
          
        """
        n = []
        for s in self.stochastics:
            n.append(s.trace.length())
        n = set(n)
        if len(n) > 1:
            raise ValueError('Traces do not have the same length.')
        elif n == 0:
            raise AttributeError('Stochastic has no trace to compute covariance.')
        else:
            n = n.pop()
            
        if type(trace) is not slice:
            trace = slice(trace, n)
            
        a = self.trace2array(trace)
        
        return np.cov(a, rowvar=0)

    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in bool_dtypes:
                raise ValueError('Binary stochastics not supported by AdaptativeMetropolis.')
            else:
                self.isdiscrete[stochastic] = False


    def dimension(self):
        """Compute the dimension of the sampling space and identify the slices
        belonging to each stochastic.
        """
        self.dim = 0
        self._slices = {}
        for stochastic in self.stochastics:
            if isinstance(stochastic.value, np.matrix):
                p_len = len(stochastic.value.A.ravel())
            elif isinstance(stochastic.value, np.ndarray):
                p_len = len(stochastic.value.ravel())
            else:
                p_len = 1
            self._slices[stochastic] = slice(self.dim, self.dim + p_len)
            self.dim += p_len


    def update_cov(self):
        """Recursively compute the covariance matrix for the multivariate normal
        proposal distribution.

        This method is called every self.interval once self.delay iterations
        have been performed.
        """

        scaling = (2.4)**2/self.dim # Gelman et al. 1996.
        epsilon = 1.0e-5
        chain = np.asarray(self._trace)

        # Recursively compute the chain mean
        self.C, self.chain_mean = self.recursive_cov(self.C, self._trace_count,
            self.chain_mean, chain, scaling=scaling, epsilon=epsilon)

        # Shrink covariance if acceptance rate is too small
        acc_rate = self.accepted / (self.accepted + self.rejected)
        if self.shrink_if_necessary:
            if acc_rate < .001:
                self.C *= .01
            elif acc_rate < .01:
                self.C *= .25
            if self.verbose > 1:
                if acc_rate < .01:
                    print_('\tAcceptance rate was',acc_rate,'shrinking covariance')
        self.accepted = 0.
        self.rejected = 0.

        if self.verbose > 1:
            print_("\tUpdating covariance ...\n", self.C)
            print_("\tUpdating mean ... ", self.chain_mean)

        # Update state
        adjustmentwarning = '\n'+\
        'Covariance was not positive definite and proposal_sd cannot be computed by \n'+ \
        'Cholesky decomposition. The next jumps will be based on the last \n' + \
        'valid covariance matrix. This situation may have arisen because no \n' + \
        'jumps were accepted during the last `interval`. One solution is to \n' + \
        'increase the interval, or specify an initial covariance matrix with \n' + \
        'a smaller variance. For this simulation, each time a similar error \n' + \
        'occurs, proposal_sd will be reduced by a factor .9 to reduce the \n' + \
        'jumps and increase the likelihood of accepted jumps.'

        try:
            self.updateproposal_sd()
        except np.linalg.LinAlgError:
            warnings.warn(adjustmentwarning)
            self.covariance_adjustment(.9)

        self._trace_count += len(self._trace)
        self._trace = []

    def covariance_adjustment(self, f=.9):
        """Multiply self.proposal_sd by a factor f. This is useful when the current proposal_sd is too large and all jumps are rejected.
        """
        self.proposal_sd *= f

    def updateproposal_sd(self):
        """Compute the Cholesky decomposition of self.C."""
        self.proposal_sd = np.linalg.cholesky(self.C)

    def recursive_cov(self, cov, length, mean, chain, scaling=1, epsilon=0):
        r"""Compute the covariance recursively.

        Return the new covariance and the new mean.

        .. math::
            C_k & = \frac{1}{k-1} (\sum_{i=1}^k x_i x_i^T - k\bar{x_k}\bar{x_k}^T)
            C_n & = \frac{1}{n-1} (\sum_{i=1}^k x_i x_i^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)
                & = \frac{1}{n-1} ((k-1)C_k + k\bar{x_k}\bar{x_k}^T + \sum_{i=k+1}^n x_i x_i^T - n\bar{x_n}\bar{x_n}^T)

        :Parameters:
            -  cov : matrix
                Previous covariance matrix.
            -  length : int
                Length of chain used to compute the previous covariance.
            -  mean : array
                Previous mean.
            -  chain : array
                Sample used to update covariance.
            -  scaling : float
                Scaling parameter
            -  epsilon : float
                Set to a small value to avoid singular matrices.
        """
        n = length + len(chain)
        k = length
        new_mean = self.recursive_mean(mean, length, chain)

        t0 = k * np.outer(mean, mean)
        t1 = np.dot(chain.T, chain)
        t2 = n*np.outer(new_mean, new_mean)
        t3 = epsilon * np.eye(cov.shape[0])

        new_cov =  (k-1)/(n-1.)*cov + scaling/(n-1.) * (t0 + t1 - t2 + t3)
        return new_cov, new_mean

    def recursive_mean(self, mean, length, chain):
        r"""Compute the chain mean recursively.

        Instead of computing the mean :math:`\bar{x_n}` of the entire chain,
        use the last computed mean :math:`bar{x_j}` and the tail of the chain
        to recursively estimate the mean.

        .. math::
            \bar{x_n} & = \frac{1}{n} \sum_{i=1}^n x_i
                      & = \frac{1}{n} (\sum_{i=1}^j x_i + \sum_{i=j+1}^n x_i)
                      & = \frac{j\bar{x_j}}{n} + \frac{\sum_{i=j+1}^n x_i}{n}

        :Parameters:
            -  mean : array
                Previous mean.
            -  length : int
                Length of chain used to compute the previous mean.
            -  chain : array
                Sample used to update mean.
        """
        n = length + len(chain)
        return length * mean / n + chain.sum(0)/n


    def propose(self):
        """
        This method proposes values for stochastics based on the empirical
        covariance of the values sampled so far.

        The proposal jumps are drawn from a multivariate normal distribution.
        """

        arrayjump = np.dot(self.proposal_sd, np.random.normal(size=self.proposal_sd.shape[0]))
        if self.verbose > 2:
            print_('Jump :', arrayjump)

        # Update each stochastic individually.
        for stochastic in self.stochastics:
            jump = arrayjump[self._slices[stochastic]]
            if np.iterable(stochastic.value):
                jump = np.reshape(arrayjump[self._slices[stochastic]],np.shape(stochastic.value))
            if self.isdiscrete[stochastic]:
                jump = round_array(jump)
            stochastic.value = stochastic.value + jump

    def step(self):
        """
        Perform a Metropolis step.

        Stochastic parameters are block-updated using a multivariate normal
        distribution whose covariance is updated every self.interval once
        self.delay steps have been performed.

        The AM instance keeps a local copy of the stochastic parameter's trace.
        This trace is used to computed the empirical covariance, and is
        completely independent from the Database backend.

        If self.greedy is True and the number of iterations is smaller than
        self.delay, only accepted jumps are stored in the internal
        trace to avoid computing singular covariance matrices.
        """

        # Probability and likelihood for stochastic's current value:
        logp = self.logp_plus_loglike
        if self.verbose > 1:
            print_('Current value: ', self.stoch2array())
            print_('Current likelihood: ', logp)

        # Sample a candidate value
        self.propose()

        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stochastic's proposed value:
            logp_p = self.logp_plus_loglike
            if self.verbose > 2:
                print_('Current value: ', self.stoch2array())
                print_('Current likelihood: ', logp_p)

            if np.log(random()) < logp_p - logp:
                accept = True
                self.accepted += 1
                if self.verbose > 2:
                    print_('Accepted')
            else:
                self.rejected += 1
                if self.verbose > 2:
                    print_('Rejected')
        except ZeroProbability:
            self.rejected += 1
            logp_p = None
            if self.verbose > 2:
                    print_('Rejected with ZeroProbability Error.')

        if (not self._current_iter % self.interval) and self.verbose > 1:
            print_("Step ", self._current_iter)
            print_("\tLogprobability (current, proposed): ", logp, logp_p)
            for stochastic in self.stochastics:
                print_("\t", stochastic.__name__, stochastic.last_value, stochastic.value)
            if accept:
                print_("\tAccepted\t*******\n")
            else:
                print_("\tRejected\n")
            print_("\tAcceptance ratio: ", self.accepted/(self.accepted+self.rejected))

        if self._current_iter == self.delay:
            self.greedy = False

        if not accept:
            self.reject()

        if accept or not self.greedy:
            self.internal_tally()

        if self._current_iter>self.delay and self._current_iter%self.interval==0:
           self.update_cov()

        self._current_iter += 1

    # Please keep reject() factored out- helps RandomRealizations figure out what to do.
    def reject(self):
        for stochastic in self.stochastics:
            # stochastic.value = stochastic.last_value
            stochastic.revert()

    def internal_tally(self):
        """Store the trace of stochastics for the computation of the covariance.
        This trace is completely independent from the backend used by the
        sampler to store the samples."""
        chain = []
        for stochastic in self.stochastics:
            chain.append(np.ravel(stochastic.value))
        self._trace.append(np.concatenate(chain))

    def trace2array(self, sl):
        """Return an array with the trace of all stochastics, sliced by sl."""
        chain = []
        for stochastic in self.stochastics:
            tr = stochastic.trace.gettrace(slicing=sl)
            if tr is None:
                raise AttributeError
            chain.append(tr)
        return np.hstack(chain)

    def stoch2array(self):
        """Return the stochastic objects as an array."""
        a = np.empty(self.dim)
        for stochastic in self.stochastics:
            a[self._slices[stochastic]] = stochastic.value
        return a


    def tune(self, verbose=0):
        """Tuning is done during the entire run, independently from the Sampler
        tuning specifications. """
        return False


class TWalk(StepMethod):
    """
    The t-walk is a scale-independent, adaptive MCMC algorithm for arbitrary
    continuous distributions and correltation structures. The t-walk maintains 
    two independent points in the sample space, and moves are based on 
    proposals that are accepted or rejected with a standard M-H acceptance 
    probability on the product space. The t-walk is strictly non-adaptive on 
    the product space, but displays adaptive behaviour on the original state 
    space. There are four proposal distributions (walk, blow, hop, traverse) 
    that together offer an algorithm that is effective in sampling 
    distributions of arbitrary scale.
    
    The t-walk was devised by J.A. Christen and C. Fox (2010).
    
    :Parameters:
      - stochastic : Stochastic
          The variable over which self has jurisdiction.
      - kernel_probs (optional) : iterable
          The probabilities of choosing each kernel.
      - walk_theta (optional) : float
          Parameter for the walk move. Christen and Fox recommend
          values in [0.3, 2] (Defaults to 1.5).
      - traverse_theta (optional) : float
          Parameter for the traverse move. Christen and Fox recommend
          values in [2, 10] (Defaults to 6.0).
      - n1 (optional) : integer
          The number of elements to be moved at each iteration.
          Christen and Fox recommend values in [2, 20] (Defaults to 4).
      - support (optional) : function
          Function defining the support of the stochastic 
          (Defaults to real line).
      - verbose (optional) : integer
          Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
      - tally (optional) : bool
          Flag for recording values for trace (Defaults to True).
    """
    def __init__(self, stochastic, inits=None, kernel_probs=[0.4918, 0.4918, 0.0082, 0.0082], walk_theta=1.5, traverse_theta=6.0, n1=4, support=lambda x: True, verbose=-1, tally=True):
        
        # Initialize superclass
        StepMethod.__init__(self, [stochastic], verbose=verbose, tally=tally)
        
        # Ordered list of proposal kernels
        self.kernels = [self.walk, self.traverse, self.blow, self.hop]
        
        # Kernel for current iteration
        self.current_kernel = None
        
        self.accepted = zeros(len(kernel_probs))
        self.rejected = zeros(len(kernel_probs))
        
        # Cumulative kernel probabilities
        self.cum_probs = np.cumsum(kernel_probs)
        
        self.walk_theta = walk_theta
        self.traverse_theta = traverse_theta
        
        # Set public attributes
        self.stochastic = stochastic
        if verbose > -1:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose
        
        # Determine size of stochastic
        if isinstance(self.stochastic.value, ndarray):
            self._len = len(self.stochastic.value.ravel())
        else:
            self._len = 1
        
        # Create attribute for holding value and secondary value
        self.values = [self.stochastic.value]
        
        # Initialize to different value from stochastic or supplied values
        if inits is None:
            self.values.append(self.stochastic.random())
            # Reset original value
            self.stochastic.value = self.values[0]
        else:
            self.values.append(inits)
        
        # Flag for using second point in log-likelihood calculations
        self._prime = False
        
        # Proposal adjustment factor for current iteration
        self.hastings_factor = 0.0
        
        # Set probability of selecting any parameter
        self.p = 1.*min(self._len, n1)/self._len
        
        # Support function
        self._support = support
        
        self._state = ['accepted', 'rejected', 'p']
        
    def n1():
        doc = "Mean number of parameters to be selected for updating"
        def fget(self):
            return self._n1
        def fset(self, value):
            self._n1 = value
            self._calc_p()
        return locals()
    n1 = property(**n1())
    
    @staticmethod
    def competence(stochastic):
        """
        The competence function for TWalk.
        """
        # if stochastic.dtype in float_dtypes and np.alen(stochastic.value) > 4:
        #             if np.alen(stochastic.value) >=10:
        #                 return 2
        #             return 1
        return 0
    
    def walk(self):
        """Walk proposal kernel"""
        
        if self.verbose>1:
            print_('\t' + self._id + ' Running Walk proposal kernel')
        
        # Mask for values to move
        phi = self.phi
        
        theta = self.walk_theta
        
        u = random(len(phi))
        z = (theta / (1 + theta))*(theta*u**2 + 2*u - 1)

        if self._prime:
            xp, x = self.values
        else:
            x, xp = self.values
            
        if self.verbose>1:
            print_('\t' + 'Current value = ' + str(x))
        
        x = x + phi*(x - xp)*z
        
        if self.verbose>1:
            print_('\t' + 'Proposed value = ' + str(x))
        
        self.stochastic.value = x
        
        # Set proposal adjustment factor
        self.hastings_factor = 0.0
    
    def traverse(self):
        """Traverse proposal kernel"""
        
        if self.verbose>1:
            print_('\t' + self._id + ' Running Traverse proposal kernel')
        
        # Mask for values to move
        phi = self.phi
        
        theta = self.traverse_theta
        
        # Calculate beta
        if (random() < (theta-1)/(2*theta)):
            beta = exp(1/(theta + 1)*log(random()))
        else:
            beta = exp(1/(1 - theta)*log(random()))
        
        if self._prime:
            xp, x = self.values
        else:
            x, xp = self.values
            
        if self.verbose>1:
            print_('\t' + 'Current value = ' + str(x))
        
        x = (xp + beta*(xp - x))*phi + x*(phi==False)
        
        if self.verbose>1:
            print_('\t' + 'Proposed value = ' + str(x))
        
        self.stochastic.value = x
    
        # Set proposal adjustment factor
        self.hastings_factor = (sum(phi) - 2)*log(beta)
    
    def blow(self):
        """Blow proposal kernel"""
        
        if self.verbose>1:
            print_('\t' + self._id + ' Running Blow proposal kernel')
        
        # Mask for values to move
        phi = self.phi
        
        if self._prime:
            xp, x = self.values
        else:
            x, xp = self.values
            
        if self.verbose>1:
            print_('\t' + 'Current value ' + str(x))
        
        sigma = max(phi*abs(xp - x))

        x = x + phi*sigma*rnormal()
        
        if self.verbose>1:
            print_('\t' + 'Proposed value = ' + str(x))
        
        self.hastings_factor = self._g(x, xp, sigma) - self._g(self.stochastic.value, xp, sigma)

        self.stochastic.value = x

    def _g(self, h, xp, s):
        """Density function for blow and hop moves"""
        
        nphi = sum(self.phi)
        
        return (nphi/2.0)*log(2*pi) + nphi*log(s) + 0.5*sum((h - xp)**2)/(s**2)
    
    
    def hop(self):
        """Hop proposal kernel"""
        
        if self.verbose>1:
            print_('\t' + self._id + ' Running Hop proposal kernel')
        
        # Mask for values to move
        phi = self.phi
        
        if self._prime:
            xp, x = self.values
        else:
            x, xp = self.values
    
        if self.verbose>1:
            print_('\t' + 'Current value of x = ' + str(x))
        
        sigma = max(phi*abs(xp - x))/3.0

        x = (xp + sigma*rnormal())*phi + x*(phi==False)
        
        if self.verbose>1:
            print_('\t' + 'Proposed value = ' + str(x))
        
        self.hastings_factor = self._g(x, xp, sigma) - self._g(self.stochastic.value, xp, sigma)
        
        self.stochastic.value = x

    
    def reject(self):
        """Sets current s value to the last accepted value"""
        self.stochastic.revert()
        
        # Increment rejected count
        self.rejected[self.current_kernel] += 1
        
        if self.verbose>1:
            print_(self._id, "rejected, reverting to value =", self.stochastic.value)
    
    def propose(self):
        """This method is called by step() to generate proposed values"""
        
        # Generate uniform variate to choose kernel
        self.current_kernel = sum(self.cum_probs < random())
        kernel = self.kernels[self.current_kernel]
        
        # Parameters to move
        self.phi = (random(self._len) < self.p)

        # Propose new value
        kernel()

    
    def step(self):
        """Single iteration of t-walk algorithm"""
                
        valid_proposal = False
        
        # Use x or xprime as pivot
        self._prime = (random() < 0.5)
        
        if self.verbose>1:
            print_("\n\nUsing x%s as pivot" % (" prime"*self._prime or ""))
        
        if self._prime:
            # Set the value of the stochastic to the auxiliary
            self.stochastic.value = self.values[1]
            
            if self.verbose>1:
                print_(self._id, "setting value to auxiliary", self.stochastic.value)
        
        # Current log-probability
        logp = self.logp_plus_loglike
        if self.verbose>1:
            print_("Current logp", logp)
        
        try:
            # Propose new value
            while not valid_proposal:
                self.propose()
                # Check that proposed value lies in support
                valid_proposal = self._support(self.stochastic.value)
                
            if not sum(self.phi):
                raise ZeroProbability

            # Proposed log-probability
            logp_p = self.logp_plus_loglike
            if self.verbose>1:
                print_("Proposed logp", logp_p)
            
        except ZeroProbability:
            
            # Reject proposal
            if self.verbose>1:
                print_(self._id + ' rejecting due to ZeroProbability.')
            self.reject()
            
            if self._prime:
                # Update value list
                self.values[1] = self.stochastic.value
                # Revert to stochastic's value for next iteration
                self.stochastic.value = self.values[0]
            
                if self.verbose>1:
                    print_(self._id, "reverting stochastic to primary value", self.stochastic.value)
            else:
                # Update value list
                self.values[0] = self.stochastic.value

            if self.verbose>1:
                print_(self._id + ' returning.')
            return
        
        if self.verbose>1:
            print_('logp_p - logp: ', logp_p - logp)
        
        # Evaluate acceptance ratio
        if log(random()) > (logp_p - logp + self.hastings_factor):
            
            # Revert s if fail
            self.reject()

        else:
            # Increment accepted count
            self.accepted[self.current_kernel] += 1
            if self.verbose > 1:
                print_(self._id + ' accepting')
        
        if self._prime:
            # Update value list
            self.values[1] = self.stochastic.value
            # Revert to stochastic's value for next iteration
            self.stochastic.value = self.values[0]
            
            if self.verbose>1:
                print_(self._id, "reverting stochastic to primary value", self.stochastic.value)
                
        else:
            # Update value list
            self.values[0] = self.stochastic.value


class IIDSStepper(StepMethod):
    """
    See written documentation.
    """
    pass
