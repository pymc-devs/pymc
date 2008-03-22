__author__ = 'Christopher Fonnesbeck, fonnesbeck@maths.otago.ac.nz'

from pymc.StepMethods import *

class Slice(StepMethod):
    """
    Samples from a posterior distribution by sampling uniformly from the region under the 
    density function. The slice sampler alternates between sampling vertically in the density and 
    sampling horizontally, from the current vertical position.
    
    This may be easier to imlpement than Gibbs Sampling, but more efficient than M-H sampling.
    """
    pass