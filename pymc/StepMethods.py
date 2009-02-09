from __future__ import division

import numpy as np
from utils import msqrt, check_type, round_array, float_dtypes, integer_dtypes, bool_dtypes, safe_len, find_generations
from numpy import ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs, exp, where, isscalar, iterable
from numpy.linalg.linalg import LinAlgError
from numpy.random import randint, random
from numpy.random import normal as rnormal
from numpy.random import poisson as rpoisson
from PyMCObjects import Stochastic, Potential, Deterministic
from Container import Container
from Node import ZeroProbability, Node, Variable, StochasticBase
from pymc.decorators import prop
from copy import copy
from InstantiationDecorators import deterministic
import pdb, warnings, sys
import inspect

__docformat__='reStructuredText'

# Changeset history
# 22/03/2007 -DH- Added a _state attribute containing the name of the attributes that make up the state of the step method, and a method to return that state in a dict. Added an id.
# TODO: Test cases for binary and discrete Metropolises.

conjugate_Gibbs_competence = 0
nonconjugate_Gibbs_competence = 0

class AdaptationError(ValueError): pass

__all__=['DiscreteMetropolis', 'Metropolis', 'StepMethod', 'assign_method',  'pick_best_methods', 'StepMethodRegistry', 'NoStepper', 'BinaryMetropolis', 'AdaptiveMetropolis','Gibbs','conjugate_Gibbs_competence', 'nonconjugate_Gibbs_competence', 'DrawFromPrior']

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
#             print '\n\tWarning, there was an error while step method %s assessed its competence \n \
# \tto handle stochastic %s. It is being excluded from consideration.\n' \
#                     %(method.__name__, stochastic)
            competence = 0

        # If better than current best method, promote it
        if competence > max_competence:
            best_candidates = set([method])
            max_competence = competence

        # If same competence, add it to the set of best methods
        elif competence == max_competence:
            best_candidates.add(method)

    if max_competence<=0:
        raise ValueError, 'Maximum competence reported for stochastic %s is <= 0... you may need to write a custom step method class.' % stochastic.__name__

    # print s.__name__ + ': ', best_candidates, ' ', max_competence
    return best_candidates

def assign_method(stochastic, scale=None):
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
            out = method(stochastic, scale = scale)
        else:
            out = method(stochastic)
    except:
        a,b,c = sys.exc_info()
        raise a, failure_header + b.message, c
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
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

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

    __metaclass__ = StepMethodMeta

    def __init__(self, variables, verbose=0):
        # StepMethod initialization

        if not iterable(variables) or isinstance(variables, Node):
            variables = [variables]

        self.stochastics = set()
        self.children = set()
        self.parents = set()

        # Initialize hidden attributes
        self.adaptive_scale_factor = 1.
        self.accepted = 0.
        self.rejected = 0.
        self._state = ['rejected', 'accepted', 'adaptive_scale_factor']
        self.verbose = verbose

        # File away the variables
        for variable in variables:
            # Sort.

            if isinstance(variable,Stochastic) and not variable.observed:
                self.stochastics.add(variable)

        if len(self.stochastics)==0:
            raise ValueError, 'No stochastics provided.'

        # Find children, no need to find parents; each variable takes care of those.
        for variable in variables:
            self.children |= variable.children
            for parent in variable.parents.itervalues():
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

        # Verbose feedback
        if verbose > 0 or self.verbose > 0:
            print '\t%s tuning:' % self._id

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
        # Warning: self.stochastic is not defined above. The following assumes
        # that the class has created this value, which is a bit fragile. DH
        if verbose > 0 or self.verbose > 0:
            if hasattr(self, 'stochastic'):
                print '\t\tvalue:', self.stochastic.value
            print '\t\tacceptance rate:', acc_rate
            print '\t\tadaptive scale factor:', self.adaptive_scale_factor
            print

        return tuning

    def _get_loglike(self):
        # Fetch log-probability (as sum of childrens' log probability)

        # Initialize sum
        sum = 0.

        # Loop over children
        for child in self.children:

            # Verbose feedback
            if self.verbose > 1:
                print '\t'+self._id+' Getting log-probability from child ' + child.__name__

            # Increment sum
            sum += child.logp
        if self.verbose > 1:
            print '\t' + self._id + ' Current log-likelihood ', sum
        return sum

    # Make get property for retrieving log-probability
    loglike = property(fget = _get_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, with self.stochastics removed.")

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
            The proposal distribution. May be 'Normal', 'RoundedNormal', 'Bernoulli',
            'Prior' or None. If None is provided, a proposal distribution is chosen
            by examining P.value's type.

    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high

    :SeeAlso: StepMethod, Sampler.
    """

    def __init__(self, stochastic, scale=1., proposal_sd=None, proposal_distribution=None, verbose=0):
        # Metropolis class initialization

        # Initialize superclass
        StepMethod.__init__(self, [stochastic], verbose=verbose)

        # Set public attributes
        self.stochastic = stochastic
        self.verbose = verbose

        # Add proposal_sd to state
        self._state += ['proposal_sd', 'proposal_distribution']

        # Avoid zeros when setting proposal variance
        if proposal_sd is not None:
            self.proposal_sd = proposal_sd
        else:
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

        # If no dist argument is provided, assign a proposal distribution automatically.
        if not proposal_distribution:

            # Pick Gaussian by default
            self.proposal_distribution = "Normal"

        else:
            self.proposal_distribution = proposal_distribution

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

        if self.verbose > 0:
            print
            print self._id + ' getting initial prior.'

        if self.proposal_distribution == "Prior":
            # No children
            logp = 0.
        else:
            logp = self.stochastic.logp

        if self.verbose > 0:
            print self._id + ' getting initial likelihood.'
        loglike = self.loglike

        if self.verbose > 0:
            print self._id + ' proposing.'

        # Sample a candidate value
        self.propose()

        # Probability and likelihood for s's proposed value:
        try:
            if self.proposal_distribution == "Prior":
                logp_p = 0.
                # Check for weirdness before accepting jump
                self.stochastic.logp
            else:
                logp_p = self.stochastic.logp
            loglike_p = self.loglike

        except ZeroProbability:

            # Reject proposal
            if self.verbose > 0:
                print self._id + ' rejecting due to ZeroProbability.'
            self.reject()

            # Increment rejected count
            self.rejected += 1

            if self.verbose > 1:
                print self._id + ' returning.'
            return

        if self.verbose > 1:
            print 'logp_p - logp: ', logp_p - logp
            print 'loglike_p - loglike: ', loglike_p - loglike

        HF = self.hastings_factor()

        # Evaluate acceptance ratio
        if log(random()) > logp_p + loglike_p - logp - loglike + HF:

            # Revert s if fail
            self.reject()

            # Increment rejected count
            self.rejected += 1
            if self.verbose > 0:
                print self._id + ' rejecting'
        else:
            # Increment accepted count
            self.accepted += 1
            if self.verbose > 0:
                print self._id + ' accepting'

        if self.verbose > 1:
            print self._id + ' returning.'

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
            self.stochastic.value = rnormal(self.stochastic.value, self.adaptive_scale_factor * self.proposal_sd)
        elif self.proposal_distribution == "Prior":
            self.stochastic.random()


class Gibbs(Metropolis):
    """
    Base class for the Gibbs step methods
    """
    def __init__(self, stochastic, verbose=0):
        Metropolis.__init__(self, stochastic, verbose=verbose)

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

            if log(np.random.random()) > logp_p - logp:
                self.reject()

    def tune(self, *args, **kwargs):
        return False

    def propose(self):
        raise NotImplementedError, 'The Gibbs class has to be subclassed, it is not usable directly.'


class DrawFromPrior(StepMethod):
    """
    Handles dataless submodels.
    """
    def __init__(self, variables, generations, verbose=0):
        StepMethod.__init__(self, variables, verbose)
        self.generations = generations

    def step(self):
        for generation in self.generations:
            for s in generation:
                s.rand()

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

    def __init__(self, stochastic, scale=1., proposal_sd=None, proposal_distribution=None, positive=False):
        # DiscreteMetropolis class initialization

        # Initialize superclass
        Metropolis.__init__(self, stochastic, scale=scale, proposal_sd=proposal_sd, proposal_distribution=proposal_distribution)

        # Initialize verbose feedback string
        self._id = stochastic.__name__

        # Flag for positive-only values
        self._positive = positive

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



class BinaryMetropolis(Metropolis):
    """
    Like Metropolis, but with a modified step() method.
    Good for binary variables.

    """

    def __init__(self, stochastic, p_jump=.1, proposal_distribution=None, verbose=0):
        # BinaryMetropolis class initialization

        # Initialize superclass
        Metropolis.__init__(self, stochastic, proposal_distribution=proposal_distribution, verbose=verbose)

        self._state.remove('proposal_sd')

        # Initialize verbose feedback string
        self._id = stochastic.__name__

        # adaptive_scale_factor controls the jump probability
        self.adaptive_scale_factor = log(1.-p_jump) / log(.5)

    @staticmethod
    def competence(stochastic):
        """
        The competence function for Binary One-At-A-Time Metropolis
        """
        if stochastic.dtype in bool_dtypes:
            return 1
        else:
            return 0

    def step(self):
        if not isscalar(self.stochastic.value):
            Metropolis.step(self)
        else:

            # See what log-probability of True is.
            self.stochastic.value = True

            try:
                logp_true = self.stochastic.logp
                loglike_true = self.loglike
            except ZeroProbability:
                self.stochastic.value = False
                return

            # See what log-probability of False is.
            self.stochastic.value = False

            try:
                logp_false = self.stochastic.logp
                loglike_false = self.loglike
            except ZeroProbability:
                self.stochastic.value = True
                return

            # Test
            p_true = exp(logp_true + loglike_true)
            p_false = exp(logp_false + loglike_false)

            if self.verbose>1:
                print """%s step information:
    - logp_true: %f
    - loglike_true: %f
    - logp_false: %f
    - loglike_false: %f
    - p_true: %f
    - p_false: %f
                """ % (self._id, logp_true, loglike_true, logp_false, loglike_false, p_true, p_false)

            # Stochastically set value according to relative
            # probabilities of True and False
            if random() > p_false / (p_true + p_false):
                if self.verbose > 0:
                    print "%s setting %s's value to True." % (self._id, self.stochastic)
                self.stochastic.value = True
            elif self.verbose > 0:
                print "%s setting %s's value to False." % (self._id, self.stochastic)


    def propose(self):
        # Propose new values

        if self.proposal_distribution == 'Prior':
            self.stochastic.random()
        else:
            # Convert adaptive_scale_factor to a jump probability
            p_jump = 1.-.5**self.adaptive_scale_factor

            rand_array = random(size=shape(self.stochastic.value))
            new_value = copy(self.stochastic.value)
            switch_locs = where(rand_array<p_jump)
            new_value[switch_locs] = True - new_value[switch_locs]
            # print switch_locs, rand_array, new_value, self.stochastic.value
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
          Initial guess for the covariance matrix C.

      - delay : int
          Number of steps before the empirical covariance is computed. If greedy
          is True, the algorithm waits for delay accepted steps before computing
          the covariance.

      - scales : dict
          Dictionary containing the scale for each stochastic keyed by name.
          If cov is None, those scales are used to define an initial covariance
          matrix. If neither cov nor scale is given, the initial covariance is
          guessed from the objects value (or trace if available).

      - interval : int
          Interval between covariance updates.

      - greedy : bool
          If True, only the accepted jumps are tallied in the internal trace
          until delay is reached. This is useful to make sure that the empirical
          covariance has a sensible structure.

      - shrink_if_necessary : bool
          If True, the acceptance rate is checked when the step method tunes. If
          the acceptance rate is small, the proposal covariance is shrunk according
          to the folllowing rule:

          if acc_rate < .001:
              self.C *= .01
          elif acc_rate < .01:
              self.C *= .25

      - verbose : int
          Controls the verbosity level.


    :Reference:
      Haario, H., E. Saksman and J. Tamminen, An adaptive Metropolis algorithm,
          Bernouilli, vol. 7 (2), pp. 223-242, 2001.
    """
    def __init__(self, stochastic, cov=None, delay=1000, scales=None, interval=200, greedy=True, shrink_if_necessary=False, verbose=0):

        # Verbosity flag
        self.verbose = verbose

        if not np.iterable(stochastic) or isinstance(stochastic, Variable):
            stochastic = [stochastic]
        # Initialize superclass
        StepMethod.__init__(self, stochastic, verbose)

        self._id = 'AdaptiveMetropolis_'+'_'.join([p.__name__ for p in self.stochastics])
        # State variables used to restore the state in a latter session.
        self._state += ['_trace_count', '_current_iter', 'C', 'proposal_sd',
        '_proposal_deviate', '_trace']

        self.proposal_sd = None

        # Number of successful steps before the empirical covariance is computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy

        # Call methods to initialize
        self.check_type()
        self.dimension()
        self.set_cov(cov, scales)
        self.updateproposal_sd()

        # Keep track of the internal trace length
        # It may be different from the iteration count since greedy
        # sampling can be done during warm-up period.
        self._trace_count = 0
        self._current_iter = 0

        self._proposal_deviate = np.zeros(self.dim)
        self.chain_mean = np.asmatrix(np.zeros(self.dim))
        self._trace = []

        if self.verbose >= 1:
            print "Initialization..."
            print 'Dimension: ', self.dim
            print "C_0: ", self.C
            print "Sigma: ", self.proposal_sd


    @staticmethod
    def competence(stochastic):
        """
        The competence function for AdaptiveMetropolis.
        The AM algorithm is well suited to deal with multivariate
        parameters.
        """
        if not stochastic.dtype in float_dtypes and not stochastic.dtype in integer_dtypes:
            return 0
            # Algorithm is not well-suited to sparse datasets. Dont use if less than
            # 25 percent of values are nonzero
        if np.alen(stochastic.value) == 1:
            return 0
        elif np.alen(stochastic.value) < 5:
            return 2
        elif (len(stochastic.value.nonzero()[0]) > 0.25*len(stochastic.value)):
            return 2
        else:
            return 0


    def set_cov(self, cov=None, scales={}, trace=2000, scaling=50):
        """Define C, the jump distributioin covariance matrix.

        Return:
            - cov,  if cov != None
            - covariance matrix built from the scales dictionary if scales!=None
            - covariance matrix estimated from the stochastics last trace values.
            - covariance matrix estimated from the stochastics value, scaled by
                scaling parameter.
        """

        if cov is not None:
            self.C = cov
        elif scales:
            # Get array of scales
            ord_sc = self.order_scales(scales)
            # Scale identity matrix
            self.C = np.eye(self.dim)*ord_sc
        else:
            try:
                a = self.trace2array(-trace, -1)
                nz = a[:, 0]!=0
                self.C = np.cov(a[nz, :], rowvar=0)
            except:
                ord_sc = []
                for s in self.stochastics:
                    this_value = abs(np.ravel(s.value))
                    if not this_value.any():
                        this_value = [1.]
                    for elem in this_value:
                        ord_sc.append(elem)
                # print len(ord_sc), self.dim
                for i in xrange(len(ord_sc)):
                    if ord_sc[i] == 0:
                        ord_sc[i] = 1
                self.C = np.eye(self.dim)*ord_sc/scaling


    def check_type(self):
        """Make sure each stochastic has a correct type, and identify discrete stochastics."""
        self.isdiscrete = {}
        for stochastic in self.stochastics:
            if stochastic.dtype in integer_dtypes:
                self.isdiscrete[stochastic] = True
            elif stochastic.dtype in bool_dtypes:
                raise 'Binary stochastics not supported by AdaptativeMetropolis.'
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


    def order_scales(self, scales):
        """Define an array of scales to build the initial covariance.
        If init_scales is None, the scale is taken to be the initial value of
        the stochastics.
        """
        ord_sc = []
        for stochastic in self.stochastics:
            ord_sc.append(np.ravel(scales[stochastic]))
        ord_sc = np.concatenate(ord_sc)

        if np.squeeze(ord_sc).shape[0] != self.dim:
            raise "Improper initial scales, dimension don't match", \
                (np.squeeze(ord_sc), self.dim)
        return ord_sc

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
        if acc_rate < .001:
            self.C *= .01
        elif acc_rate < .01:
            self.C *= .25
        if self.verbose > 0:
            if acc_rate < .01:
                print '\tAcceptance rate was',acc_rate,'shrinking covariance'
        self.accepted = 0.
        self.rejected = 0.

        if self.verbose > 0:
            print "\tUpdating covariance ...\n", self.C
            print "\tUpdating mean ... ", self.chain_mean

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
            print 'Jump :', arrayjump

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
        logp = sum([stochastic.logp for stochastic in self.stochastics])
        loglike = self.loglike
        if self.verbose > 1:
            print 'Current value: ', self.stoch2array()
            print 'Current likelihood: ', logp+loglike

        # Sample a candidate value
        self.propose()

        # Metropolis acception/rejection test
        accept = False
        try:
            # Probability and likelihood for stochastic's proposed value:
            logp_p = sum([stochastic.logp for stochastic in self.stochastics])
            loglike_p = self.loglike
            if self.verbose > 2:
                print 'Current value: ', self.stoch2array()
                print 'Current likelihood: ', logp+loglike

            if np.log(random()) < logp_p + loglike_p - logp - loglike:
                accept = True
                self.accepted += 1
                if self.verbose > 2:
                    print 'Accepted'
            else:
                self.rejected += 1
                if self.verbose > 2:
                    print 'Rejected'
        except ZeroProbability:
            self.rejected += 1
            logp_p = None
            loglike_p = None
            if self.verbose > 2:
                    print 'Rejected with ZeroProbability Error.'

        if (not self._current_iter % self.interval) and self.verbose > 1:
            print "Step ", self._current_iter
            print "\tLogprobability (current, proposed): ", logp, logp_p
            print "\tloglike (current, proposed):      : ", loglike, loglike_p
            for stochastic in self.stochastics:
                print "\t", stochastic.__name__, stochastic.last_value, stochastic.value
            if accept:
                print "\tAccepted\t*******\n"
            else:
                print "\tRejected\n"
            print "\tAcceptance ratio: ", self.accepted/(self.accepted+self.rejected)

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

    def trace2array(self, i0, i1):
        """Return an array with the trace of all stochastics from index i0 to i1."""
        chain = []
        for stochastic in self.stochastics:
            chain.append(ravel(stochastic.trace.gettrace(slicing=slice(i0,i1))))
        return concatenate(chain)

    def stoch2array(self):
        """Return the stochastic objects as an array."""
        a = np.empty(self.dim)
        for stochastic in self.stochastics:
            a[self._slices[stochastic]] = stochastic.value
        return a


    def tune(self, verbose):
        """Tuning is done during the entire run, independently from the Sampler
        tuning specifications. """
        return False

class IIDSStepper(StepMethod):
    """
    See written documentation.
    """
    pass
