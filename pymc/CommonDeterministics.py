from PyMCObjects import *
from distributions import *
from InstantiationDecorators import deterministic
import numpy as np
import inspect
from utils import safe_len

__all__ = ['CompletedDirichlet', 'LinearCombination', 'Index', 'Lambda', 'lambda_deterministic', 'lam_dtrm']

class Lambda(Deterministic):
    """
    L = Lambda(name, lambda p1=p1, p2=p2: f(p1, p2)[, dtype=None, trace=True, cache_depth=2, plot=True, verbose=0])
    
    Converts second argument, an anonymous function, into
    Deterministic object with specified name.
    """
    def __init__(self, name, lam_fun, doc='A Deterministic made from an anonymous function', *args, **kwds):
        
            (parent_names, junk0, junk1, parent_values) = inspect.getargspec(lam_fun)

            if junk0 is not None \
              or junk1 is not None \
              or parent_values is None:
                raise ValueError, '%s: All arguments to lam_fun must have default values.' % name

            if not len(parent_names) == len(parent_values):
                raise ValueError, '%s: All arguments to lam_fun must have default values.' % name

            parents = dict(zip(parent_names[-len(parent_values):], parent_values))

            Deterministic.__init__(self, eval=lam_fun, name=name, parents=parents, doc=doc, *args, **kwds)
            
lambda_deterministic = Lambda
lam_dtrm = Lambda


class CompletedDirichlet(Deterministic):
    """
    CD = CompletedDirichlet(name, D[, trace=True, cache_depth=2, plot=True, verbose=0])
    
    'Completes' the value of D by appending 1-sum(D.value) to the end.
    """
    def __init__(self, name, D, trace=True, cache_depth=2, plot=True, verbose=0):
        
        def eval_fun(D):
            N = D.shape[1]
            out = np.empty((1,N+1))
            out[0,:N] = D
            out[0,N] = 1.-np.sum(D)
            return out
        
        Deterministic.__init__(self, eval=eval_fun, name=name, parents={'D': D}, doc='The completed version of %s'%D.__name__,
         dtype=float, trace=trace, cache_depth=cache_depth, plot=plot, verbose=verbose)        
        
        
class LinearCombination(Deterministic):
    """
    L = LinearCombination(name, x, y[, dtype=None, trace=True, cache_depth=2, plot=True, verbose=0])

    A Deterministic returning the sum of dot(x[i],y[i]).

    x and y must be lists or single Stochastics.
    
    Attributes:
     - x
     - y
     - N: length of x and y
     - coefs: dictionary keyed by stochastic indicating what each stochastic is multiplied by.
     - sides:                   "                       whether each stochastic is in x or y.
     - offsets:                 "                       everything that gets added to each stochastic and its coefficient.
    """
    
    def __init__(self, name, x, y, doc = 'A linear combination of several variables', *args, **kwds):
        self.x = x
        self.y = y
        self.N = len(self.x)

        if not len(self.y)==len(self.x):
            raise ValueError, 'Arguments x and y must be same length.'

        def eval_fun(x, y):
            out = np.dot(x[0], y[0])
            for i in xrange(1,len(x)):
                out = out + np.dot(x[i], y[i])
            return out

        Deterministic.__init__(self,
                                eval=eval_fun,
                                doc=doc,
                                name = name,
                                parents = {'x':x, 'y':y},
                                *args, **kwds)

        # Tabulate coefficients and offsets of each constituent Stochastic.                        
        self.coefs = {}
        self.sides = {}

        for s in self.parents.stochastics | self.parents.data_stochastics:
            self.coefs[s] = []
            self.sides[s] = []

        for i in xrange(self.N):

            stochastic_elem = None

            if isinstance(x[i], Stochastic):

                if x[i] is y[i]:
                    raise ValueError, 'Stochastic %s multiplied by itself in LinearCombination %s.' %(x[i], self)

                stochastic_elem = x[i]
                self.sides[stochastic_elem].append('L')                
                this_coef = Lambda('%s_coef'%stochastic_elem, lambda c=y[i]: np.asarray(c))
                self.coefs[stochastic_elem].append(this_coef)

            if isinstance(y[i], Stochastic):
                
                stochastic_elem = y[i]
                self.sides[stochastic_elem].append('R')                
                this_coef = Lambda('%s_coef'%stochastic_elem, lambda c=x[i]: np.asarray(c))
                self.coefs[stochastic_elem].append(this_coef)

                
        self.sides = pymc.Container(self.sides)
        self.coefs = pymc.Container(self.coefs)


class Index(LinearCombination):
    """
    I = Index(name, x, y, index[, dtype=None, trace=True, cache_depth=2, plot=True, verbose=0])

    A deterministic returning dot(x[index], y[index]).

    Useful for hierarchical models/ clustering/ discriminant analysis.
    Emulates LinearCombination to make it easier to write Gibbs step
    methods that can deal with such cases.

    Attributes:
      - index
      - x, y, N, coefs, offsets, sides: Same as LinearCombination, but with x = x[index] and y = 1.
    """
    def __init__(self, name, x, index, doc = "Selects one of a list of several variables", *args, **kwds):
        self.index = Lambda('index', lambda i=index: np.int(i))
        self.N = 1
        self.x = x
        self.y = []

        def eval_fun(x, index):
            return x[index]

        Deterministic.__init__(self,
                                eval=eval_fun,
                                doc=doc,
                                name = name,
                                parents = {'x':x, 'index':self.index},
                                *args, **kwds)

        # Tabulate coefficients and offsets of each constituent Stochastic.                        
        self.coefs = {}
        self.sides = {}

        for s in self.parents.stochastics | self.parents.data_stochastics:
            
            @deterministic
            def coef(index=self.index, x=x):
                if self.x[index] is s:
                    return np.eye(safe_len(x[index]))
                else:
                    return None
                                
            self.coefs[s] = [coef]
            self.sides[s] = ['L']
     
        self.sides = pymc.Container(self.sides)
        self.coefs = pymc.Container(self.coefs)
