.. _prob_dists:

*************************
Probability Distributions
*************************

The most fundamental step in building Bayesian models is the specification of a full probability model for the problem at hand. This primarily involved assigning parametric statistical distributions to unknown quantities in the model, in addition to appropriate functional forms for likelihoods to represent the information from the data. To this end, PyMC3 includes a comprehensive set of pre-defined statistical distribution that can be used as model building blocks. 


::

    class SomeDistribution(Continuous):
    
        def random(self):
            ...
            return random_samples
            
        def logp(self, value):
            ...
            return total_log_prob
            



Custom Distributions
====================


            
Auto-transformation
===================
