__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

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
    Updates stochastics of base distribution and concentration stoch.

    Very easy: likelihood is DPDraws' logp,
    propose a new DP to match.
    """
    pass
    
class DPDrawMetropolis(StepMethod):
    """
    Updates DP draws.
    """
    pass
