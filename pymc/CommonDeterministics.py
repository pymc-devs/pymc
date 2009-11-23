"""
pymc.CommonDeterministics

A collection of Deterministic subclasses to handle common situations.
It's a good idea to use these rather than user-defined objects when
possible, as some fitting methods (particularly Gibbs step methods)
will know how to handle them but not user-defined objects with
equivalent functionality.
"""

__docformat__='reStructuredText'

import PyMCObjects as pm
from Node import Variable
from Container import Container
from InstantiationDecorators import deterministic
import numpy as np
import inspect
from utils import safe_len, stukel_logit, stukel_invlogit, logit, invlogit
from types import UnboundMethodType
from copy import copy
import sys

__all__ = ['CompletedDirichlet', 'LinearCombination', 'Index', 'Lambda', 'lambda_deterministic', 'lam_dtrm',
            'logit', 'invlogit', 'stukel_logit', 'stukel_invlogit', 'Logit', 'InvLogit', 'StukelLogit', 'StukelInvLogit',
            'pfunc']#+['iter_','complex_','int_','long_','float_','oct_','hex_']

class Lambda(pm.Deterministic):
    """
    L = Lambda(name, lambda p1=p1, p2=p2: f(p1, p2)[,
        doc, dtype=None, trace=True, cache_depth=2, plot=None])

    Converts second argument, an anonymous function, into a
    Deterministic object with specified name.

    :Parameters:
      name : string
        The name of the deteriministic object to be created.
      lambda : function
        The function from which the deterministic object should
        be created. All arguments must be given default values!
      p1, p2, ... : any
        The parameters of lambda.
      other parameters :
        See docstring of Deterministic.

    :Note:
      Will work even if argument 'lambda' is a named function
      (defined using def)

    :SeeAlso:
      Deterministic, Logit, StukelLogit, StukelInvLogit, Logit, InvLogit,
      LinearCombination, Index
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

            pm.Deterministic.__init__(self, eval=lam_fun, name=name, parents=parents, doc=doc, *args, **kwds)

def lambda_deterministic(*args, **kwargs):
    """
    An alias for Lambda

    :SeeAlso:
      Lambda
    """
    return Lambda(*args, **kwargs)

def lam_dtrm(*args, **kwargs):
    """
    An alias for Lambda

    :SeeAlso:
      Lambda
    """
    return Lambda(*args, **kwargs)

class Logit(pm.Deterministic):
    """
    L = Logit(name, theta[, doc, dtype=None, trace=True,
        cache_depth=2, plot=None])

    A deterministic variable whose value is the logit of parent theta.

    :Parameters:
      name : string
        The name of the variable.
      theta : number, array or variable
        The parent to which the logit function should be applied.
        Must be between 0 and 1.
      other parameters :
        See docstring of Deterministic.

    :SeeAlso:
      Deterministic, Lambda, InvLogit, StukelLogit, StukelInvLogit
    """
    def __init__(self, name, theta, doc='A logit transformation', *args, **kwds):
        pm.Deterministic.__init__(self, eval=logit, name=name, parents={'theta': theta}, doc=doc, *args, **kwds)


class InvLogit(pm.Deterministic):
    """
    P = InvLogit(name, ltheta[, doc, dtype=None, trace=True,
        cache_depth=2, plot=None])

    A Deterministic whose value is the inverse logit of parent ltheta.

    :Parameters:
      name : string
        The name of the variable.
      ltheta : number, array or variable
        The parent to which the inverse logit function should be
        applied.
      other parameters :
        See docstring of Deterministic.

    :SeeAlso:
      Deterministic, Lambda, Logit, StukelLogit, StukelInvLogit
    """
    def __init__(self, name, ltheta, doc='An inverse logit transformation', *args, **kwds):
        pm.Deterministic.__init__(self, eval=invlogit, name=name, parents={'ltheta': ltheta}, doc=doc, *args, **kwds)


class StukelLogit(pm.Deterministic):
    """
    S = StukelLogit(name, theta, a1, a2, [, doc, dtype=None, trace=True,
        cache_depth=2, plot=None])

    A Deterministic whose value is Stukel's link function with
    parameters a1 and a2 applied to theta.

    To see the effects of a1 and a2, try plotting the function stukel_logit
    on theta=linspace(.1,.9,100)

    :Parameters:
      name : string
        The name of the variable.
      theta : number, array or variable.
        The parent to which the link function should be
        applied. Must be between 0 and 1.
      a1 : number
        One of the shape parameters.
      a2 : number
        The other shape parameter.
      other parameters :
        See docstring of Deterministic.

    :Reference:
      Therese A. Stukel, 'Generalized Logistic Models',
      JASA vol 83 no 402, pp.426-431 (June 1988)

    :SeeAlso:
      Deterministic, Lambda, Logit, InvLogit, StukelInvLogit
    """
    def __init__(self, name, theta, a1, a2, doc="Stukel's link function", *args, **kwds):
        pm.Deterministic.__init__(self, eval=stukel_logit,
                    name=name, parents={'theta': theta, 'a1': a1, 'a2': a2},
                    doc=doc, *args, **kwds)


class StukelInvLogit(pm.Deterministic):
    """
    P = StukelInvLogit(name, ltheta, a1, a2, [, doc, dtype=None,
        trace=True, cache_depth=2, plot=None])

    A Deterministic whose value is Stukel's inverse link function with
    parameters a1 and a2 applied to ltheta.

    To see the effects of a1 and a2, try plotting the function stukel_invlogit
    on ltheta=linspace(-5,5,100)

    :Parameters:
      name : string
        The name of the variable.
      ltheta : number, array or variable.
        The parent to which the inverse link function should
        be applied. Must be between 0 and 1.
      a1 : number
        One of the shape parameters.
      a2 : number
        The other shape parameter.
      other parameters :
        See docstring of Deterministic.

    :Reference:
      Therese A. Stukel, 'Generalized Logistic Models',
      JASA vol 83 no 402, pp.426-431 (June 1988)

    :SeeAlso:
      Deterministic, Lambda, Logit, InvLogit, StukelLogit
    """
    def __init__(self, name, ltheta, a1, a2, doc="Stukel's inverse link function", *args, **kwds):
        pm.Deterministic.__init__(self, eval=stukel_invlogit,
                    name=name, parents={'ltheta': ltheta, 'a1': a1, 'a2': a2},
                    doc=doc, *args, **kwds)


class CompletedDirichlet(pm.Deterministic):
    """
    CD = CompletedDirichlet(name, D[, doc, trace=True,
        cache_depth=2, plot=None])

    'Completes' the value of D by appending 1-sum(D.value) to the end.

    :Parameters:
      name : string
        The name of the variable.
      D : array or variable
        Value of object will be 1-sum(D) or 1-sum(D.value).
        Sum of D or D's value must be between 0 and 1.
      other parameters:
        See docstring of Deterministic

    :SeeAlso:
      Deterministic, Lambda, Index, LinearCombination
    """
    def __init__(self, name, D, doc=None, trace=True, cache_depth=2, plot=None, verbose=None):

        def eval_fun(D):
            N = len(D)
            out = np.empty((1,N+1))
            out[0,:N] = D
            out[0,N] = 1.-np.sum(D)
            return out

        if doc is None:
            doc = 'The completed version of %s'%D.__name__

        pm.Deterministic.__init__(self, eval=eval_fun, name=name, parents={'D': D}, doc=doc,
         dtype=float, trace=trace, cache_depth=cache_depth, plot=plot, verbose=verbose)


class LinearCombination(pm.Deterministic):
    """
    L = LinearCombination(name, x, y[, doc, dtype=None,
        trace=True, cache_depth=2, plot=None])

    A Deterministic returning the sum of dot(x[i],y[i]).

    :Parameters:
      name : string
        The name of the variable
      x : list or variable
        Will be multiplied against y and summed.
      y : list or variable
        Will be multiplied against x and summed.
      other parameters :
        See docstring of Deterministic.

    :Attributes:
      x : list or variable
        Input argument
      y : list or variable
        Input argument
      N : integer
        length of x and y
      coefs : dictionary
        Keyed by variable. Indicates what each variable is multiplied by.
      sides : dictionary
        Keyed by variable. Indicates whether each variable is in x or y.
      offsets : dictionary
        Keyed by variable. Indicates everything that gets added to each
        stochastic and its coefficient.

    :SeeAlso:
      Deterministic, Lambda, Index
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
            return np.asarray(out).squeeze()

        pm.Deterministic.__init__(self,
                                eval=eval_fun,
                                doc=doc,
                                name = name,
                                parents = {'x':x, 'y':y},
                                *args, **kwds)

        # Tabulate coefficients and offsets of each constituent Stochastic.
        self.coefs = {}
        self.sides = {}

        for s in self.parents.stochastics | self.parents.observed_stochastics:
            self.coefs[s] = []
            self.sides[s] = []

        for i in xrange(self.N):

            stochastic_elem = None

            if isinstance(x[i], pm.Stochastic):

                if x[i] is y[i]:
                    raise ValueError, 'Stochastic %s multiplied by itself in LinearCombination %s.' %(x[i], self)

                stochastic_elem = x[i]
                self.sides[stochastic_elem].append('L')
                this_coef = Lambda('%s_coef'%stochastic_elem, lambda c=y[i]: np.asarray(c))
                self.coefs[stochastic_elem].append(this_coef)

            if isinstance(y[i], pm.Stochastic):

                stochastic_elem = y[i]
                self.sides[stochastic_elem].append('R')
                this_coef = Lambda('%s_coef'%stochastic_elem, lambda c=x[i]: np.asarray(c))
                self.coefs[stochastic_elem].append(this_coef)


        self.sides = Container(self.sides)
        self.coefs = Container(self.coefs)


class Index(LinearCombination):
    """
    I = Index(name, x, y, index[, doc, dtype=None, trace=True,
        cache_depth=2, plot=None])

    A deterministic returning dot(x[index], y[index]).

    Useful for hierarchical models/ clustering/ discriminant analysis.
    Emulates LinearCombination to make it easier to write Gibbs step
    methods that can deal with such cases.

    :Parameters:
      name : string
        The name of the variable
      x : list or variable
        Will be multiplied against y and summed.
      y : list or variable
        Will be multiplied against x and summed.
      index : integer or variable
        Index to use when computing value.
      other parameters :
        See docstring of Deterministic.

    :Attributes:
      index : variable
        Valued as current index.
      x, y, N, coefs, offsets, sides :
        Same as LinearCombination, but with x = x[index] and y = 1.

    :SeeAlso:
      Deterministic, Lambda, LinearCombination

    TODO: Special lazy function for Index that only caches value of parent currently
    'pointed to'.
    """
    def __init__(self, name, x, index, doc = "Selects one of a list of several variables", *args, **kwds):
        self.index = Lambda('index', lambda i=index: np.int(i))
        self.N = 1
        self.x = x
        self.y = []

        def eval_fun(x, index):
            return x[index]

        pm.Deterministic.__init__(self,
                                eval=eval_fun,
                                doc=doc,
                                name = name,
                                parents = {'x':x, 'index':self.index},
                                *args, **kwds)

        # Tabulate coefficients and offsets of each constituent Stochastic.
        self.coefs = {}
        self.sides = {}

        for s in self.parents.stochastics | self.parents.observed_stochastics:

            @deterministic
            def coef(index=self.index, x=x):
                if self.x is s:
                    return np.eye(safe_len(x[index]))
                elif self.x[index] is s:
                    return np.eye(safe_len(x[index]))
                else:
                    return None

            self.coefs[s] = [coef]
            self.sides[s] = ['L']

        self.sides = Container(self.sides)
        self.coefs = Container(self.coefs)

# TODO: Write tests for all this!

# =================================================================
# = pfunc converts ordinary functions to Deterministic factories. =
# =================================================================
def pufunc(func):
    """
    Called by pfunc to convert NumPy ufuncs to deterministic factories.
    """
    def dtrm_generator(*args):
        if len(args) != func.nin:
            raise ValueError, 'invalid number of arguments'
        name = func.__name__ + '('+'_'.join([str(arg) for arg in list(args)])+')'
        doc_str = 'A deterministic returning %s(%s)'%(func.__name__, ', '.join([str(arg) for arg in args]))
        parents = {}
        for i in xrange(func.nin):
            parents['in%i'%i] = args[i]
        def wrapper(**kwargs):
            return func(*[kwargs['in%i'%i] for i in xrange(func.nin)])
        return pm.Deterministic(wrapper, doc_str, name, parents, trace=False, plot=False)
    dtrm_generator.__name__ = func.__name__ + '_deterministic_generator'
    dtrm_generator.__doc__ = """
Deterministic-generating wrapper for %s. Original docstring:
%s

%s
    """%(func.__name__, '_'*60, func.__doc__)
    return dtrm_generator


def pfunc(func):
    """
    pf = pfunc(func)

    Returns a function that can be called just like func; however its arguments may be
    PyMC objects or containers of PyMC objects, and its return value will be a deterministic.

    Example:

        >>> A = pymc.Normal('A',0,1,size=10)
        >>> pprod = pymc.pfunc(numpy.prod)
        >>> B = pprod(A, axis=0)
        >>> B
        <pymc.PyMCObjects.Deterministic 'prod(A_0)' at 0x3ce49b0>
        >>> B.value
        -0.0049333289649554912
        >>> numpy.prod(A.value)
        -0.0049333289649554912
    """
    if isinstance(func, np.ufunc):
        return pufunc(func)
    elif not inspect.isfunction(func):
        if func.__name__ == '__call__':
            raise ValueError, 'Cannot get argspec of call method. Is it builtin?'
        try:
            return pfunc(func.__call__)
        except:
            cls, inst, tb = sys.exc_info()
            inst = cls('Failed to create pfunc wrapper from object %s. Original error message:\n\n%s'%(func, inst.message))
            raise cls, inst, tb
    fargs, fvarargs, fvarkw, fdefaults = inspect.getargspec(func)
    n_fargs = len(fargs)

    def dtrm_generator(*args, **kwds):
        name = func.__name__ + '('+'_'.join([str(arg) for arg in list(args) + kwds.values()])+')'
        doc_str = 'A deterministic returning %s(%s, %s)'%(func.__name__, ', '.join([str(arg) for arg in args]), ', '.join(['%s=%s'%(key, str(val)) for key, val in kwds.iteritems()]))

        parents = {}
        varargs = []
        for kwd, val in kwds.iteritems():
            parents[kwd] = val
        for i in xrange(len(args)):
            if i < n_fargs:
                parents[fargs[i]] = args[i]
            else:
                varargs.append(args[i])

        if len(varargs)==0:
            eval_fun = func
        else:
            parents['varargs']=varargs
            def wrapper(**wkwds_in):
                wkwds = copy(wkwds_in)
                wargs = []
                for arg in fargs:
                    wargs.append(wkwds.pop(arg))
                wargs.extend(wkwds.pop('varargs'))
                return func(*wargs, **wkwds)
            eval_fun = wrapper

        return pm.Deterministic(eval_fun, doc_str, name, parents, trace=False, plot=False)
    dtrm_generator.__name__ = func.__name__ + '_deterministic_generator'
    dtrm_generator.__doc__ = """
Deterministic-generating wrapper for %s. Original docstring:
%s

%s
    """%(func.__name__, '_'*60, func.__doc__)
    return dtrm_generator


# ==========================================================
# = Add special methods to variables to support FBC syntax =
# ==========================================================

def create_uni_method(op_name, klass):
    """
    Creates a new univariate special method, such as A.__neg__() <=> -A,
    for target class. The method is called __op_name__.
    """
    # This function will become the actual method.
    def new_method(self):
        # This code creates a Deterministic object.
        def eval_fun(self,op=op):
            return getattr(self, op)()
        return pm.Deterministic(eval_fun,
                                'A Deterministic returning the value of %s(%s)'%(op_name, self.__name__),
                                '('+op_name+'_'+self.__name__+')',
                                {'self':self, 'op': '__'+op_name+'__'},
                                trace=False,
                                plot=False)
    # Make the function into a method for klass.
    new_method.__name__ = '__'+op_name+'__'
    setattr(klass, new_method.__name__, UnboundMethodType(new_method, None, klass))

def create_casting_method(op, klass):
    """
    Creates a new univariate special method, such as A.__float__() <=> float(A.value),
    for target class. The method is called __op_name__.
    """
    # This function will become the actual method.
    def new_method(self, op=op):
        return op(self.value)
    # Make the function into a method for klass.
    new_method.__name__ = '__'+op.__name__+'__'
    setattr(klass, new_method.__name__, UnboundMethodType(new_method, None, klass))

def create_rl_bin_method(op_name, klass):
    """
    Creates a new binary special method with left and right versions, such as
        A.__mul__(B) <=> A*B,
        A.__rmul__(B) <=> [B*A if B.__mul__(A) fails]
    for target class. The method is called __op_name__.
    """
    # Make left and right versions.
    for prefix in ['r','']:
        # This function will became the methods.
        def new_method(self, other, prefix=prefix):
            # This code will create one of two Deterministic objects.
            if prefix == 'r':
                # Right version: raises error on failure.
                parents = {'self':self, 'other':other, 'op':'__r' + op_name + '__'}
                def eval_fun(self,other,op):
                    out = getattr(self, op)(other)
                    if out is NotImplemented:
                        # the rt version has failed, meainng the lft version has failed as well.
                        raise TypeError, "unsupported operand type(s) for %s: '%s' and '%s'"%(op.replace('_',''), self.__class__.__name__, other.__class__.__name__)
                    return out
            else:
                # Left version: tries right version on failure.
                parents = {'self':self, 'other':other, 'op':'__' + op_name + '__', 'rt_op': '__r'+op_name+'__'}
                def eval_fun(self, other, op, rt_op):
                    out = getattr(self, op)(other)
                    if out is NotImplemented:
                        # if try the rt version.
                        out = getattr(other, rt_op)(self)
                    return out
            return pm.Deterministic(eval_fun,
                                    'A Deterministic returning the value of %s(%s,%s)'%(prefix+op_name,self.__name__, str(other)),
                                    '('+'_'.join([self.__name__,prefix+op_name,str(other)])+')',
                                    parents,
                                    trace=False,
                                    plot=False)
        # Convert the functions into methods for klass.
        new_method.__name__ = '__'+prefix+op_name+'__'
        setattr(klass, new_method.__name__, UnboundMethodType(new_method, None, klass))


def create_rl_lin_comb_method(op_name, klass, x_roles, y_roles):
    """
    Creates a new binary special method with left and right versions, such as
        A.__mul__(B) <=> A*B,
        A.__rmul__(B) <=> [B*A if B.__mul__(A) fails]
    for target class. The method is called __op_name__.
    """
    # This function will became the methods.
    def new_method(self, other, x_roles=x_roles, y_roles=y_roles):
        x = []
        y = []
        for xr in x_roles:
            if xr=='self':
                x.append(self)
            elif xr=='other':
                x.append(other)
            else:
                x.append(xr)
        for yr in y_roles:
            if yr=='self':
                y.append(self)
            elif yr=='other':
                y.append(other)
            else:
                y.append(yr)
        # This code will create one of two Deterministic objects.
        return LinearCombination('('+'_'.join([self.__name__,op_name,str(other)])+')', x, y, trace=False, plot=False)

    # Convert the functions into methods for klass.
    new_method.__name__ = '__'+op_name+'__'
    setattr(klass, new_method.__name__, UnboundMethodType(new_method, None, klass))

def create_bin_method(op_name, klass):
    """
    Creates a new binary special method with only a left version, such as
    A.__eq__(B) <=> A==B, for target class. The method is called __op_name__.
    """
    # This function will become the method.
    def new_method(self, other):
        # This code creates a Deterministic object.
        def eval_fun(self, other, op):
            return getattr(self, op)(other)
        return pm.Deterministic(eval_fun,
                                'A Deterministic returning the value of %s(%s,%s)'%(op_name,self.__name__, str(other)),
                                '('+'_'.join([self.__name__,op_name,str(other)])+')',
                                {'self':self, 'other':other, 'op':'__'+op_name+'__'},
                                trace=False,
                                plot=False)
    # Convert the function into a method for klass.
    new_method.__name__ = '__'+op_name+'__'
    setattr(klass, new_method.__name__, UnboundMethodType(new_method, None, klass))


# Left/right binary operators
for op in ['div', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or']:
    create_rl_bin_method(op, Variable)

# # Binary operators
# for op in ['lt', 'le', 'eq', 'ne', 'gt', 'ge']:
#     create_bin_method(op ,Variable)

# Unary operators
for op in ['unicode','neg','pos','abs','invert','index']:
    create_uni_method(op, Variable)

# Casting operators
for op in [iter,complex,int,long,float,oct,hex,len]:
    create_casting_method(op, Variable)

# Addition, subtraction, multiplication
# TODO: Uncomment once LinearCombination issues are ironed out.
# create_rl_lin_comb_method('add', Variable, ['self', 'other'], [1,1])
# create_rl_lin_comb_method('radd', Variable, ['self', 'other'], [1,1])
# create_rl_lin_comb_method('sub', Variable, ['self', 'other'], [1,-1])
# create_rl_lin_comb_method('rsub', Variable, ['self', 'other'], [-1,1])
# create_rl_lin_comb_method('mul', Variable, ['self'],['other'])
# create_rl_lin_comb_method('rmul', Variable, ['self'],['other'])
#TODO: Comment once LinearCombination issues are ironed out.
for op in ['add', 'mul', 'sub']:
    create_rl_bin_method(op, Variable)


# Create __getitem__ method.
def __getitem__(self, index):
    # If index is number or number-valued variable, make an Index object
    name = '('+'_'.join([self.__name__,'getitem',str(index)])+')'
    if isinstance(index, Variable):
        if np.isscalar(index.value):
            if np.isreal(index.value):
                return Index(name, self, index)
    elif np.isscalar(index):
        if np.isreal(index):
            return Index(name, self, index, trace=False, plot=False)
    # Otherwise make a standard Deterministic.
    def eval_fun(self, index):
        return self[index]
    return pm.Deterministic(eval_fun,
                            'A Deterministic returning the value of %s(%s,%s)'%('getitem',self.__name__, str(index)),
                            name,
                            {'self':self, 'index':index},
                            trace=False,
                            plot=False)
Variable.__getitem__ = UnboundMethodType(__getitem__, None, Variable)

# Create __call__ method for Variable.
def __call__(self, *args, **kwargs):
    def eval_fun(self, args=args, kwargs=kwargs):
        return self(*args, **kwargs)
    return pm.Deterministic(eval_fun,
                            'A Deterministic returning the value of %s(*%s, **%s)'%(self.__name__, str(args), str(kwargs)),
                            self.__name__+'(*%s, **%s)'%(str(args), str(kwargs)),
                            {'self':self, 'args': args, 'kwargs': kwargs},
                            trace=False,
                            plot=False)
Variable.__call__ = UnboundMethodType(__call__, None, Variable)


# These are not working
# nonworking_ops = ['iter','complex','int','long','float','oct','hex','coerce','contains','len']
# These should NOT be implemented because they are in-place updates.
do_not_implement_ops = ['iadd','isub','imul','itruediv','ifloordiv','imod','ipow','ilshift','irshift','iand','ixor','ior']