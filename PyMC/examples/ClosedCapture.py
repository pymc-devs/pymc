"""
Closed-population capture-mark-recapture (CMR) estimation using MCMC with PyMC.
"""

from PyMC2 import MetropolisHastings

class M0 (MetropolisHastings):
    # Null capture model: no behavioural, temporal or individual heterogeneity effects

    def __init__(self, captures):
        # Class initialization

        # Initialize superclass
        MetropolisHastings.__init__(self)
        
        # Capture histories
        self.captures = captures

        # Capture probability
        self.parameter('p', init_val = 0.5)
        
        # Population estimate
        self.node('N')


    def calculate_likelihood(self):
        # Log-likelihood

        # Local variables
        captures = self.captures

        # Initialize N
        self.Nhat = 0.0

        # Initialize likelihood
        like = 0.0

        # Loop over capture histories
        for hist in captures:
            #capture frequencies of ith history
            z = data[hist]
            f.append([z])
            pp = 1.
            pbar = 1.
            for j, x in enumerate(hist):
                #cast to integer
                x = int(x)
                pp *=  x * self.p or (1. - x) * (1. - self.p)
                pbar *=  (1. - self.p)
            #probability of history
            pi += [pp]

            #weighted sum by ch frequencies
            self.Nhat +=  z / (1. - pbar)     

        #drop last history (gotten by subtraction from M)
        f = f[: -1]
        #drop last history (gotten by subtraction from 1)
        pi = pi[: -1]


        like += self.multinomial_like(f, sum(f), pi)      

        return like
