"""
Dirichlet process classes:

- DP: A Dirichlet process realization. 
    
    Attributes:
    - atoms: A list containing the atom locations.
    
    Methods:
    - __call__(m): A function returning m random values from the distribution.
    
DPParameter: A parameter valued as a DP realization. Is this necessary?

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
        -   nu: The whatever parameter.
        -   n, atoms (optional): Can be initialized conditional on previous draws. 
            Useful for Gibbs sampling, maybe MH too.
        -   basemeas_params: The parameters of the base measure.        
    """


    def __init__(self, basemeas, nu, atoms = [], n = [], **basemeas_params):
        
        super(DP, self).__init__()
        
        # The base measure and its parameters.
        self.basemeas = basemeas
        self.basemeas_params = basemeas_params

        # The whatever parameter.
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
                new_val = self.basemeas(**self.basemeas_params)
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
        
        
    
        