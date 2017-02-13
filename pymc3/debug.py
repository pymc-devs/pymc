from .blocking import DictToVarBijection

# TODO I could not locate this function used anywhere in the code base
# do we need it?


def eval_univariate(f, var, idx, point, x):
    """
    Evaluate a function as a at a specific point and only varying values at one index.

    Useful for debugging misspecified likelihoods.

    Parameters
    ----------

    f : function : dict -> val
    var : variable
    idx : index into variable
    point : point at which to center
    x : array points at which to evaluate x

    """
    bij = DictToVarBijection(var, idx, point)
    return list(map(bij.mapf(f), x))
