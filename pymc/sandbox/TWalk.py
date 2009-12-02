__author__ = 'Christopher Fonnesbeck, fonnesbeck@maths.otago.ac.nz'

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

    The t-walk was proposed by J.A. Christen an C. Fox (unpublished manuscript).

    :Parameters:
      - stochastic : Stochastic
          The variable over which self has jurisdiction.
      - kernel_probs (optional) : iterable
          The probabilities of choosing each kernel.
      - walk_theta (optional) : float
          Parameter for the walk move.
      - traverse_theta (optional) : float
          Paramter for the traverse move.
      - verbose (optional) : integer
          Level of output verbosity: 0=none, 1=low, 2=medium, 3=high
      - tally (optional) : bool
          Flag for recording values for trace (Defaults to True).
    """
    def __init__(self, stochastic, kernel_probs=[0.0008, 0.4914, 0.4914, 0.0082, 0.0082], walk_theta=0.5, traverse_theta=4.0, verbose=None, tally=True):

        # Initialize superclass
        StepMethod.__init__(self, [stochastic], verbose=verbose, tally=tally)

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
        pass

    def hop(self):
        """Hop proposal kernel"""
        pass

    def traverse(self):
        """Traverse proposal kernel"""
        pass

    def beta(self, a):
        """Auxiliary method for traverse proposal"""
        if (random() < (a-1)/(2*a)):
            return exp(1/(a+1)*log(random()))
        else:
            return exp(1/(1-a)*log(random()))

    def blow(self):
        """Blow proposal kernel"""
        pass
