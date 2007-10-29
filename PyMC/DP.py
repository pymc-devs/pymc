# TODO: It would be nice to have this done by the 2.0 release, as it showcases how Python's dynamic typing gives us flexibility.

"""
Dirichlet process classes:

- DP: A Dirichlet process realization. 
    
    Attributes:
    - atoms: A list containing the atom locations.
    
    Methods:
    - __call__(m): A function returning m random values from the distribution.
    
DPStochastic: A stoch valued as a DP realization. Is this necessary?

Need to write rdiscrete and discrete_like.

Consider calling 'DP' Realization, so it's DP.Realization and GP.Realization.

Neal cite: Markov chain random methods for Dirichlet process mixture models.
Also study up on Gibbs sampler.

Should be written in Pyrex once interface is settled.
"""

from numpy import array, cumsum
from numpy.random import uniform
from copy import copy

class DP(object):
    """
    A Dirichlet process realization.
    
    Arguments:
        -   basemeas: The base measure. Must be a function which, when called with argument n, returns a value.
        -   nu: The whatever stoch.
        -   n, atoms (optional): Can be initialized conditional on previous draws. 
            Useful for Gibbs sampling, maybe MH too.
        -   basemeas_stochs: The stochs of the base measure.        
    """


    def __init__(self, basemeas, nu, atoms = [], n = [], **basemeas_stochs):
        
        super(DP, self).__init__()
        
        # The base measure and its stochs.
        self.basemeas = basemeas
        self.basemeas_stochs = basemeas_stochs

        # The whatever stoch.
        self.nu = float(nu)

        # The number of draws from each atom.
        self.n = copy(n)
        
        # The values of the atoms.
        self.atoms = copy(atoms)
        
        # The stick. Write this cumulative sum in C once you go to Pyrex.
        self.stick = cumsum(self.n)
        
        # Sanity check.
        if not len(n) == len(atoms):
            raise ValueError, 'n and atoms must have same length in DP init method.'
        
    def __call__(self, m=1):

        """
        Calling a DP realization with argument m returns
        m values from the random measure.
        """
        
        val = []
        N = len(self.atoms)

        # Initialize. Optimization 1: keep running sum.
        if N>0:
            sum_n = sum(self.n)
        else:
            sum_n = 0
            
        float_sumn = float(sum_n)
        
        for i in xrange(m):
            
            # Optimization 2: update cumulative sum on the fly.
            self.stick = cumsum(self.n)
            
            # Maybe draw a new atom
            if uniform() > float_sumn / (float_sumn+self.nu):
                new_val = self.basemeas(**self.basemeas_stochs)
                self.atoms.append(new_val)
                self.n.append(1)
                N = N + 1

            # Otherwise draw from one of the existing algorithms
            else:
                # Optimization 3: Draw uniforms ahead of time.
                # DON'T use the same uniform for checking new atom
                # creation AND for finding which old atom to draw from,
                # you'll introduce painful bias.
                
                unif = uniform() * float_sumn
                for i in xrange(N):
                    if unif < self.stick[i]:
                        new_val = self.atoms[i]
                        self.n[i] = self.n[i]+1
                        break
                           
            float_sumn = float_sumn + 1.
            val.append(new_val)
            
        if m>1:
            return array(val, dtype=float)
        else:
            return val[0]
        
class DPStochastic(object):
    """
    value: A DP object.
    
    Parents: 'alpha': concentration stoch, 'base': base probability distribution.
    Base parent must have random() and logp() methods (must be an actual distribution object).
    
    No logp (error raised).
    """
    pass
    
class DPMetropolis(StepMethod):
    """
    Give DPStochastic a new value conditional on parents and children. Always Gibbs.
    Easy. The actual DPStochastic is only being traced for future convenience,
    it doesn't really get used by the step methods.
    
    You may want to leave this out of the model for simplicity, and tell
    users how to get draws for the DP given the trace.
    
    Alternatively, you could try to move the logp down here, but that
    would require a new algorithm.
    """
    pass
    
class DPParentMetropolis(StepMethod):
    """
    Updates stochs of base distribution and concentration stoch.
    
    Very easy: likelihood is DPDraws' logp,
    propose a new DP to match.
    """
    pass

class DPDraws(object):
    """
    value: An array of values, need to figure out dtype.
    N: length of value.
    
    May want to hide these in the step method, 
    but many step methods need them so it's probably better to keep them here:
    N_clusters: number of clusters.
    clusters: values of clusters, length-N list.
    cluster_multiplicities: multiplicities of clusters.
    
    Note may want to make these things their own Stochastics, in case people want to have
    Deterministics etc. depending on them or to trace them.
    
    Parent: 'dist': a DPStochastic.
    
    logp: product of base logp evaluated on each cluster (each cluster appears only once
    regardless of multiplicity) plus some function of alpha and the number of clusters.
    """
    pass
    
class DPDrawMetropolis(StepMethod):
    """
    Updates DP draws.
    """
    pass

    
"""
Note: If you could get a distribution for the multiplicities of the currently-
found clusters in a DP, could you give its children a logp attribute?

Then you could do something like with the GP: give the DPStochastic an intrinsic
set of clusters unrelated to its children, assess its logp using only its intrinsic
clusters, etc.

Yes, you can easily do this. Give the DP object its intrinsic clusters, and let the
step methods treat those as the things that are really participating in the model
even though from the user's perspective the entire DP is participating.

DUDE you can even figure out the DP object's logp attribute as you go along using the stick-breaking representation.

First thing you need to do is give the DP object a logp, and coordinate it with the random. 
Then everything else should come together.
"""