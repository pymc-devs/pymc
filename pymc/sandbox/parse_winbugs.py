"""
Syntax needed:

arrays: x[n:m], x[], x[,3] just translate straight to numpy arrays

repeated structures:
for (i in a:b) {
    list of statements to be repeated for increasing values of loop-variable i
}

Replace '.' with '_'

Note will allow some disallowed WinBugs syntax because 'compiles' to Python.

Need to deal with constructions like
z <- sqrt(y)
z <- dnorm(mu, tau)
which are allowed only if z is data and if 'sqrt' is invertible.

In datafile parser: Pass S-plus structures to rpy... but note WinBugs is row-major, S-Plus is column-major.

Need to extract model definition from compound document.

Can also write PyMC to WinBugs?
"""

import pyparsing as pp
import pymc as pm
import numpy as np


# ===========================================
# = Map BUGS arithmetic obs to numpy ufuncs =
# ===========================================
arith_ops = {'+':np.add,
            '*': np.multiply,
            '/': np.divide,
            '-': np.subtract}


# ==================================================
# = Map BUGS functions to numpy or scipy functions =
# ==================================================

bugs_funs = {'cloglog': lambda x: log(-log(1-x)),
            # 'cut' : None   Not implemented.
            'equals': np.equal,
            'inprod': pm.LinearCombination,
            'interp.lin': np.interp,
            'inverse': np.linalg.inv,
            'logdet': lambda x: np.log(np.linalg.det(x)), # Can be optimized
            'logit': pm.logit,
            'phi': pm.utils.normcdf,
            'pow': np.power,
            'rank': lambda x, s: np.sum(x<=s),
            'ranked': lambda x, s: np.ranx(x)[s],
            'round': np.round,
            'sd': np.std,
            'step': lambda x: x>0,
            'trunc': np.floor}

for numpy_synonym in ['abs', 'cos', 'exp', 'log', 'max', 'mean', 'min', 'sin', 'sqrt', 'round', 'sum']:
    bugs_funs[numpy_synonym] = getattr(np, numpy_synonym)

try:
    import scipy
    bugs_funs.update({'logfact': scipy.special.gammaln,
    'loggam': scipy.special.gammaln})
except:
    pass


# ========================================================
# = Map BUGS distributions to PyMC Stochastic subclasses =
# ========================================================

bugs_dists = {'bern': (pm.Bernoulli, 'p'),
                'bin': (pm.Binomial, 'p', 'n'),
                'cat': (pm.Categorical, 'p'),
                # 'negbin': (pm.NegativeBinomial, '') Need to implement standard parameterization or else translate with a Deterministic.
                'pois': (pm.Poisson, 'mu'),
                'beta': (pm.Beta, 'alpha', 'beta'),
                'chisqr': (pm.Chi2, 'nu'),
                # 'dexp': Double exponential distribution not implemented.
                'exp': (pm.Exponential, 'beta'),
                'gamma': (pm.Gamma, 'alpha', 'beta'),
                # 'gen.gamma': Not implemented
                'lnorm': (pm.Lognormal, 'mu', 'tau'),
                # 'logis': Logistic distribution not implemented
                'norm': (pm.Normal, 'mu', 'tau'),
                # 'par': Pareto distribution not implemented.
                # 't': T distribution not implemented !?
                'unif': (pm.Uniform, 'lower', 'upper'),
                # 'weib': Uses different parameterization than we do.
                'multi': (pm.Multinomial, 'p', 'n'),
                # 'dirch': Need to apply CompletedDirichlet
                'mnorm': (pm.MvNormal, 'mu', 'tau'),
                # 'mt': Multivariate student's T not implemented
                'wish': (pm.Wishart, 'T', 'n')}


# =====================================
# = Helper functions for BUGS grammar =
# =====================================

def check_distribution(toks):
    if toks[0] not in bugs_dists:
        raise NameError, 'Distribution "%s" has no analogue in PyMC.' % toks[0]

def check_function(toks):
    if len(toks) > 1:
        if toks[0] not in bugs_funs:
            if toks[0] == 'cut':
                raise NameError, 'Function "cut" has no analogue in PyMC.'
            elif toks[0] in ['logfact', 'loggam']:
                raise NameError, 'Function "%s" requires scipy.special.gammaln, which could not be imported.' % toks[0]
            else:
                raise NameError, 'Function %s is not provided by WinBugs.' % toks[0]


# ================
# = BUGS grammar =
# ================

sl = lambda st: pp.Literal(st).suppress()
slo = lambda st: pp.Optional(pp.Literal(st)).suppress()

BugsSlice = pp.Forward()

# Convert to reference to existing PyMC object or new array, number or Deterministic
BugsVar = pp.Word(pp.alphas).setResultsName('name') + pp.Optional(BugsSlice).setResultsName('slice')

# Convert to reference to existing object or number
BugsAtom = pp.Group((BugsVar ^ pp.Word(pp.nums)))

# Convert to number, array or exising object
BugsExpr = pp.Forward()

# Convert to numpy slice or slice-valued deterministic
Bugs1dSlice = pp.Group(pp.delimitedList(pp.Optional(BugsExpr),':'))

# Convert to tuple of whatever Bugs1dSlice is
BugsSlice << pp.Group(sl('[') + pp.delimitedList(Bugs1dSlice) + sl(']'))

# Convert to Stochastic subclass or submodel class.
BugsDistribution = (sl('d') + pp.Word(pp.alphas).setResultsName('dist') + sl('(') + pp.delimitedList(BugsExpr).setResultsName('args') + sl(')'))\
    .setParseAction(lambda s, l, toks: check_distribution(toks))

# Convert both of these to function, Deterministic instance or new array or number.
BugsFunction = pp.Group(pp.Word(pp.alphas).setResultsName('fun') + sl('(') + pp.delimitedList(BugsExpr).setResultsName('args') + sl(')'))\
                .setParseAction(lambda s, l, toks: check_function(toks))

BugsExpr << pp.operatorPrecedence(BugsFunction ^ BugsAtom,
                                    [('-', 1, pp.opAssoc.RIGHT),
                                    (pp.oneOf('* /'),2, pp.opAssoc.LEFT),
                                    (pp.oneOf('- +'), 2, pp.opAssoc.LEFT)])

# Convert to Stochastic instance or to submodel.
BugsStochastic = pp.Group(BugsVar.setResultsName('lhs') + sl('~') + BugsDistribution.setResultsName('rhs'))

# RHS should be Deterministic instance already, change it as needed.
BugsDeterministic = pp.Group(BugsVar.setResultsName('lhs') + sl('<-') + (BugsFunction | BugsExpr).setResultsName('rhs'))

# Convert to PyMC submodel.
BugsSubModel = pp.OneOrMore(BugsStochastic ^ BugsDeterministic)


# ===============================================
# = Parse actions corresponding to BUGS grammar =
# ===============================================


if __name__ == '__main__':

    st = """
    x[3] ~ dnorm(2, 5+3)
    y ~ dnorm(x[18:23,5,], 4)
    z[] <- log(exp(x+y)/k, y)
    w <- x + (y + z) * x
    """

    q=BugsSubModel.parseString(st)
