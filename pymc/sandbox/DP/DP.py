__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

"""
Dirichlet process classes:

- DPRealization: A Dirichlet process realization. Based on stick-breaking representation,
    but step methods should use other representations.
    
    Attributes:
    - atoms: A list containing the atom locations.
    
    Methods:
    - rand(m): Returns m random values.
    - logp(x): A function returning the log-probability of x.
    
DP: A stochastic valued as a DP realization.

DPDraw: A stochastic distributed as a DP object's value.

Neal cite: Markov chain random methods for Dirichlet process mixture models.
Also study up on Gibbs sampler.

This should all be written in Pyrex eventually. Many things are screaming for
optimization. The C++ vector class would be helpful too but that would have
to be swigged.
"""

import numpy as np
from copy import copy
from pymc import *



def draws_to_atoms(draws):
    """
    atoms, n = draws_to_atoms(draws)
    
    atoms is a list of the unique elements in draws,
    and n is a list of their corresponding multiplicities.
    
    Needs optimization badly I'm sure.
    """
    atoms = []
    n = []
    for element in np.atleast_1d(draws):
        match=False
        for i in xrange(len(atoms)):
            if all(element == atoms[i]):
                n[i] += 1
                match=True
                break
        if not match:
            atoms.append(element)
            n.append(1)
    return atoms, n



try:

    import pylab as pl

    def plot_atoms(DPr):
        """
        plot_atoms(DPr)

        Plots the atoms of DP realization DPr.
        Base measure must be over the real line.
        """
        for pair in zip(DPr.atoms, DPr.n):
            plot([pair[0], pair[0]], [0,pair[1]], 'k-')        

except ImportError:
    pass



class DPRealization(object):
    """
    A Dirichlet process realization. This is based on the stick-breaking representation
    rather than the Chinese restaurant process in order to provide a logp method. Step methods are
    free to use the Chinese restaurant process, though.
    
    Arguments:
        -   basemeas: The base measure. Must be a function which, when called with argument n, returns a value.
        -   nu: The whatever parameter.
        -   draws (optional): DPRealization can be initialized conditional on previous draws.
            Useful for Gibbs sampling, maybe MH too.
        -   basemeas_params: The parameters of the base measure.        
        
    Methods:
        -   rand(m): Returns m random values.
        -   logp(x): Returns the log-probability of x.
    """

    def __init__(self, basemeas_rand, nu, draws=[], **basemeas_params):

        # The base measure and its parameters.
        self.basemeas_rand = basemeas_rand
        self.basemeas_params = basemeas_params

        # The tightness parameter.
        self.nu = np.float(nu)

        if len(draws)>0:
            atoms, n = draws_to_atoms(draws)
            
            # The number of draws from each atom.
            self.n = n

            # The values of the atoms.
            self.atoms = atoms

            # Need to triple-check that this is OK!            
            # The probability masses of the atoms.
            mass_sofar = rbeta(sum(n), nu)
            if len(n) > 1:
                self.mass = list((rdirichlet(n) * mass_sofar).squeeze())
            else:
                self.mass = [mass_sofar]
            self.mass_sofar = mass_sofar
            self.mass_prod = 1.
            for m in self.mass:
                self.mass_prod *= (1.-m)
            
        else:
            self.n = []
            self.atoms = []
            self.mass = []
            self.mass_sofar = 0.
            self.mass_prod = 1.
            
    def logp(self, value):
        """
        F.logp(x)
        
        Returns the log of the probability mass assigned to x.
        Returns -Inf if x is not in self.atoms; this behavior is fine
        for continuous base distributions but incorrect for discrete.
        """
        logp_out = 0.
        value = np.atleast_1d(value)
        for val_now in value:
            match=False
            for i in xrange(len(self.atoms)):
                if all(val_now == self.atoms[i]):
                    logp_out += log(self.mass[i])
                    match=True
                    break
            if not match:
                return -Inf
        return logp_out
        
    def rand(self, m=1):
        """
        F.rand(m=1)
        
        Returns m values from the random probability distribution.
        """
        
        draws = np.empty(m, dtype=float)
        
        for i in xrange(m):
            

            # Draw from existing atoms
            if np.random.random()  < self.mass_sofar:
                atom_index = int(flib.rcat(np.asarray(self.mass) / self.mass_sofar,0,1,1))
                new_draw = self.atoms[atom_index]
                self.n[atom_index] += 1
                
            # Make new atom                    
            else:
                
                new_draw = self.basemeas_rand(**self.basemeas_params)
                self.atoms.append(new_draw)
            
                self.n.append(1)
            
                new_mass = self.mass_prod * rbeta(1, self.nu)
                self.mass.append(new_mass)
                self.mass_prod *= 1.-new_mass
                self.mass_sofar += new_mass
            
            draws[i] = new_draw
            
        if m==1:
            draws = draws[0]
        return draws
        
        
        
        
class DP(Stochastic):
    """
    value: A DP realization.
    
    Parents: 'alpha': concentration parameter, 'base': base probability distribution.
    Base parent must have random() and logp() methods (must be an actual distribution object).
    
    Should get intrinsic set of clusters. Step methods will update them with the children.
    A new value should be created conditional on the intrinsic clusters every time a parent is updated.
    """
    def __init__(self, 
                name, 
                basemeas_rand, 
                basemeas_logp, 
                nu, 
                doc=None, 
                trace=True, 
                value=None, 
                cache_depth=2, 
                plot=False,
                verbose=0,
                **basemeas_params):
        
        self.basemeas_logp = basemeas_logp
        self.basemeas_rand = basemeas_rand
        self.basemeas_params = basemeas_params
        
        parents = {}
        
        parents['basemeas_logp'] = basemeas_logp
        parents['basemeas_rand'] = basemeas_rand
        parents['basemeas_params'] = basemeas_params
        parents['nu'] = nu
        
        def dp_logp_fun(value, **parents):
            return 0.
            # raise ValueError, 'DPStochastic objects have no logp attribute'
            
        def dp_random_fun(basemeas_logp, basemeas_rand, nu, basemeas_params):
            return DPRealization(basemeas_rand, nu, **basemeas_params)
        
        # If value argument provided, read off intrinsic clusters.
        # If clusters argument provided, well store them.
        # If no clusters argument provided, propose from prior all over the place.
        
        Stochastic.__init__(self, logp=dp_logp_fun, random=dp_random_fun, doc=doc, name=name, parents=parents,
                            trace=trace, value=value, dtype=np.object, rseed=True, isdata=False, cache_depth=cache_depth,
                            plot=plot, verbose=verbose)




class DPDraw(Stochastic):
    """
    value: An array of values.
    
    May want to hide these in the step method, 
    but many step methods need them so it's probably better to keep them here:
    N: length of value.
    N_clusters: number of clusters.
    clusters: values of clusters, length-N list.
    cluster_multiplicities: multiplicities of clusters.
    
    Note may want to make these things their own Stochastics, in case people want to have
    Deterministics etc. depending on them or to trace them.
    
    Parent: 'dist': a DPStochastic.
    
    logp: product of base logp evaluated on each cluster (each cluster appears only once
    regardless of multiplicity) plus some function of alpha and the number of clusters.
    """
    def __init__(   self,
                    name,
                    DP, 
                    N=1,                    
                    doc=None,                      
                    trace=True, 
                    isdata=False,
                    cache_depth=2,
                    plot=True,
                    verbose = 0):
        
        self.N = N
        
        def DP_logp_fun(value, dist):
            return dist.logp(value)
            
        def DP_random_fun(dist):
            return dist.rand(N)
                    
        Stochastic.__init__(self, 
                            logp = DP_logp_fun, 
                            doc=doc, 
                            name=name, 
                            parents={'dist': DP}, 
                            random = DP_random_fun, 
                            trace=trace, 
                            value=None,
                            dtype=float, 
                            rseed=True, 
                            isdata=isdata,
                            cache_depth=cache_depth,
                            plot=plot,
                            verbose = verbose)                    

        self.clusters = lam_dtrm('clusters',lambda draws=self: draws_to_atoms(draws))
    













from numpy.testing import *
from pylab import *
class test_DP(NumpyTestCase):

    def check_correspondence(self):
        x_d = linspace(-5.,5.,1000)
        dx = x_d[1] - x_d[0]
        nu = 10
        
        p = nu * dx/sqrt(2.*pi)*exp(-x_d**2)
        DP_approx = rdirichlet(p).squeeze()
        DP_approx = hstack((DP_approx, 1.-sum(DP_approx)))
        
        true_DP = DPRealization(rnormal, nu, mu=0,tau=1)
        true_DP.rand(1000)
        
        clf()
        subplot(2,1,1)
        plot(x_d, DP_approx,'k.',markersize=8)
        subplot(2,1,2)
        plot_atoms(true_DP)
    
    def check_draws(self):
        D = DPRealization(rnormal,100,mu=-10,tau=.1)
        draws = D.rand(1000)
        clf()
        hist(draws)
    
    def check_stochastics(self):
        S = DP('S', rnormal,normal_like, 100, mu=10, tau=.1)
        q = DPDraw('q', S, N=1000)
        clf()
        hist(q.value)
        


if __name__=='__main__':
    NumpyTest().run()

"""
Note: If you could get a distribution for the multiplicities of the currently-
found clusters in a DP, could you give its children a logp attribute?

Then you could do something like with the GP: give the DPStochastic an intrinsic
set of clusters unrelated to its children, assess its logp using only its intrinsic
clusters, etc.

Yes, you can easily do this. Give the DP object its intrinsic clusters, and let the
step methods treat those as the things that are really participating in the model
even though from the user's perspective the entire DP is participating.
"""

# Old random method
# val = []
# N = len(self.atoms)
# 
# # Initialize. Optimization 1: keep running sum.
# if N>0:
#     sum_n = np.sum(self.n)
# else:
#     sum_n = 0
#     
# float_sumn = np.float(sum_n)
# 
# for i in xrange(m):
#     
#     # Optimization 2: update cumulative sum on the fly.
#     self.tables = np.cumsum(self.n)
#     
#     # Maybe draw a new atom
#     if uniform() > float_sumn / (float_sumn+self.nu):
#         new_val = self.basemeas_rand(**self.basemeas_params)
#         self.atoms.append(new_val)
#         self.n.append(1)
#         N = N + 1
# 
#     # Otherwise draw from one of the existing algorithms
#     else:
#         # Optimization 3: Draw uniforms ahead of time.
#         # DON'T use the same uniform for checking new atom
#         # creation AND for finding which old atom to draw from,
#         # you'll introduce painful bias.
#         
#         unif = uniform() * float_sumn
#         for i in xrange(N):
#             if unif < self.tables[i]:
#                 new_val = self.atoms[i]
#                 self.n[i] = self.n[i]+1
#                 break
#                    
#     float_sumn = float_sumn + 1.
#     val.append(new_val)
#     
# if m>1:
#     return array(val, dtype=float)
# else:
#     return val[0]
