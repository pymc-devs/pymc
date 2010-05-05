__author__ = 'Christopher Fonnesbeck, fonnesbeck@gmail.com'

from pymc.StepMethods import *

class TWalk(StepMethod):
    """
    The t-walk is a scale-independent, adaptive MCMC algorithm for arbitrary
    continuous distributions and correltation structures. The t-walk maintains two
    independent points in the sample space, and moves are based on proposals that
    are accepted or rejected with a standard M-H acceptance probability on the
    product space. The t-walk is strictly non-adaptive on the product space, but
    displays adaptive behaviour on the original state space. There are four proposal
    distributions (walk, blow, hop, traverse) that together offer an algorithm that
    is effective in sampling distributions of arbitrary scale.
    
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
      - la (optional) : float
          A parameter used to calculate the probability of choosing each
          parameter (Defaults to 0.2876821)
      - support (optional) : function
          Function defining the support of the stochastic (Defaults to real line).
      - verbose (optional) : integer
          Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
      - tally (optional) : bool
          Flag for recording values for trace (Defaults to True).
    """
    def __init__(self, stochastic, kernel_probs=[0.4918, 0.4918, 0.0082, 0.0082], walk_theta=1.5, traverse_theta=6.0, n1=4, la=0.2876821, support=lambda x: True, verbose=None, tally=True):
        
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
        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose
        
        # Determine size of stochastic
        if isinstance(self.stochastic.value, ndarray):
            self._len = len(self.stochastic.value.ravel())
        else:
            self._len = 1
            
        # Make second point from copy of stochastic
        self.stochastic_prime = copy(self.stochastic)
        self.stochastic_prime.__name__ = self.stochastic.__name__ + ' prime'
        # Don't need to store values
        self.stochastic_prime.trace = False
        # Initialize to different value from stochastic
        self.stochastic_prime.random()
        
        # Flag for using second point in log-likelihood calculations
        self._prime = False
        
        # Proposal adjustment factor for current iteration
        self.hastings_factor = 0.0
        
        # Probability of selecting any parameter
        self.p = (n1 - (n1 - 1) * exp(-la * (self._len-1))) / self._len
        
        # Support function
        self._support = support
        
        self._state = ['accepted', 'rejected', 'p']

    def _get_logp_plus_loglike(self):
        
        # Calculate log-likelihood plus current log-probability for both x and xprime
        sum = [logp_of_set(self.markov_blanket), logp_of_set(list([self.stochastic.prime])+list(self.children))]
        
        if self.verbose>1:
            print '\t' + self._id + ' Current log-likelihood plus current log-probability', sum
        return sum

    # Make get property for retrieving log-probability
    logp_plus_loglike = property(fget = _get_logp_plus_loglike, doc="The summed log-probability of all stochastic variables that depend on \n self.stochastics, and self.stochastics.")
    
    @staticmethod
    def competence(stochastic):
        """
        The competence function for TWalk.
        """
        if stochastic.dtype in integer_dtypes:
            return 0
        else:
            return 1
    
    def walk(self):
        """Walk proposal kernel"""
        
        # Mask for values to move
        phi = self.phi
        
        theta = self.walk_theta
        
        u = random(sum(phi))
        z = (theta / (1 + theta))*(theta*u**2 + 2*u - 1)
        
        x = self.stochastic.value[phi]
        xp = self.stochastic_prime.value[phi]
        
        if self._prime:
            
            self.stochastic_prime.value[phi] = xp + (xp - x)*z
            
        else:
            
            self.stochastic.value[phi] = x + (x - xp)*z
            
        # Set proposal adjustment factor
        self.hastings_factor = 0.0
    
    def traverse(self):
        """Traverse proposal kernel"""
        
        # Mask for values to move
        phi = self.phi
        
        theta = self.traverse_theta
        
        # Calculate beta
        if (random() < (theta-1)/(2*theta)):
            beta = exp(1/(theta + 1)*log(random()))
        else:
            beta = exp(1/(1 - theta)*log(random()))
            
        x = self.stochastic.value[phi]
        xp = self.stochastic_prime.value[phi]
        
        if self._prime:
            
            self.stochastic_prime.value[phi] = x + beta*(x - xp)
            
        else:
            
            self.stochastic.value[phi] = xp + beta*(xp - x)
            
        # Set proposal adjustment factor
        self.hastings_factor = (sum(phi) - 2)*log(beta)
    
    def blow(self):
        """Blow proposal kernel"""
        
        # Mask for values to move
        phi = self.phi
        
        x_all = copy(self.stochastic.value)
        x = x_all[phi]
        xp_all = copy(self.stochastic_prime.value)
        xp = xp_all[phi]
		
		if self._prime:
		    
		    sigma = max(phi*abs(x - xp))
            
            self.stochastic_prime.value[phi] = xp + sigma*rnormal()
            
            self.hastings_factor = self._gblow(self.stochastic_prime.value, xp_all, x_all) - self._gblow(xp_all, self.stochastic_prime.value, x_all)
            
        else:
            
            sigma = max(phi*abs(xp - x))
            
            self.stochastic.value[phi] = x + sigma*rnormal()
            
            self.hastings_factor = self._g(self.stochastic.value, x_all, xp_all) - self._g(x_all, self.stochastic.value, xp_all)

    
    def _g(self, h, x, xp, s):
        """Proposal for blow and hop moves"""
        
        nphi = sum(self.phi)
        
        return (nphi/2.0)*log(2*pi) + nphi*log(s) + 0.5*sum((h - xp)**2)/(s**2)


    def hop(self):
        """Hop proposal kernel"""
        
        # Mask for values to move
        phi = self.phi
        
        x_all = copy(self.stochastic.value)
        x = x_all[phi]
        xp_all = copy(self.stochastic_prime.value)
        xp = xp_all[phi]
		
		if self._prime:
		    
		    sigma = max(phi*abs(x - xp))/3.0
            
            self.stochastic_prime.value[phi] = x + sigma*rnormal()
            
            self.hastings_factor = self._gblow(self.stochastic_prime.value, xp_all, x_all) - self._gblow(xp_all, self.stochastic_prime.value, x_all)
            
        else:
            
            sigma = max(phi*abs(xp - x))/3.0
            
            self.stochastic.value[phi] = xp + sigma*rnormal()
            
            self.hastings_factor = self._g(self.stochastic.value, x_all, xp_all) - self._g(x_all, self.stochastic.value, xp_all)
            
    
    def reject(self):
        """Sets current s value to the last accepted value"""
        self.stochastic.revert()
    
    def propose(self):
        """This method is called by step() to generate proposed values"""
        
        # Generate uniform variate to choose kernel
        self.current_kernel = sum(self.cum_probs < random())
        kernel = self.kernels[self.current_kernel]
        
        # Parameters to move
		self.phi = (random(self._len) < self.p)
		
		# Use x or xprime as pivot
        self._prime = (random() < 0.5)
        
        # Propose new value
        kernel()
    
    def step(self):
        """Single iteration of t-walk algorithm"""
        
        valid_proposal = False
        
        # Current log-probability
        logp = self.logp_plus_loglike
        
        # Propose new value
        while not valid_proposal:
            self.propose()
            # Check that proposed value lies in support
            valid_proposal = self._support(self.stochastic.value)
            
        # Proposed log-probability
        logp_p = self.logp_plus_loglike
        
        if self.verbose>1:
            print 'logp_p - logp: ', logp_p[self._prime] - logp[self._prime]

        # Evaluate acceptance ratio
        if log(random()) > logp_p[self._prime] - logp[self._prime] + self.hastings_factor:

            # Revert s if fail
            self.reject()

            # Increment rejected count
            self.rejected[self.current_kernel] += 1
            if self.verbose > 1:
                print self._id + ' rejecting'
        else:
            # Increment accepted count
            self.accepted[self.current_kernel] += 1
            if self.verbose > 1:
                print self._id + ' accepting'
                
        if self.verbose > 1:
            print self._id + ' returning.'
