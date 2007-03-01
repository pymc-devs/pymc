from PyMC2.SamplingMethods import SamplingMethod

class Gibbs(SamplingMethod):
    """
    It would be nice to organize Gibbs samplers somehow,
    to keep conjugate and non-conjugate versions together.

    This could avoid some code duplication: The conjugate and
    non-conjugate versions do similar things, but the
    non-conjugate version needs to test the proposal and the
    conjugate version needs to incorporate the prior.
    
    Also, it would be nice if the various non-parent, non-child
    values used in Gibbs sampling could be either nodes, parameters,
    or just plain values without us having to write the cases in
    every single Gibbs sampler.
    """
    def step():
        for parameter in self.parameters:
            parameter.random()
        
        
        


    
