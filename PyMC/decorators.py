#-------------------------------------------------------------
# Decorators
#-------------------------------------------------------------
# TODO: Replace flat's with ravel's, and if possible avoid resize-ing (try to
# avoid any memory allocation, in fact).

import numpy as np
import proposition4
from numpy import inf, random, sqrt
import string
import inspect
import types
from distributions import *

def fortranlike_method(f, snapshot, mv=False):
    """
    Decorator function building likelihood method for Sampler
    =========================================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib to
    create a method for the Sampler class. Must be called inside the class.
    
    self: Object the method will be assigned to.
    snapshot: Snapshot of the distributions local dictionary.
    mv: multivariate (True/False)
    
    Assume args = (self, x, param1, param2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the parameters have the same shape as x.


    Add compatibility with Sampler class
    --------------------------------------------------
    * Add a name keyword.
    * Add a 'prior' keyword (True/False).
    * If self._gof is True and prior is False, return the GoF (Goodness of Fit),
    put the gof points in self._gof_loss
    * A 'loss' keyword can be given, to specify the loss function used in the 
    computation of the GoF points. 
    """
    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.
        
    # Find the functions needed to compute the gof points.
    expval_func = snapshot[name+'_expval']
    random_func = snapshot['r'+name]
    
    def wrapper(self, *args, **kwds):
        args = list(args)
        # Shape manipulations
        xshape = np.shape(args[0])
        newargs = [np.asarray(args[0])]
        for arg in args[1:]:
            newargs.append(np.resize(arg, xshape))
        
        if self._gof and not kwds.pop('prior', False):
            """Compute gof points."""               
            name = kwds.pop('name', name)
            try:    
                self._like_names.append(name)
            except AttributeError:
                pass
                         
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,self.loss)
            self._gof_loss.append(gof_points)
        
        else:
            """Return likelihood."""
            try:
                return f(*newargs, **kwds)
            except LikelihoodError:
                return -np.Inf
        

    # Assign function attributes to wrapper.
##    wrapper.__doc__ = f.__doc__+'\n'+like.__name__+\
##        '(self, '+string.join(f.func_code.co_varnames, ', ')+\
##        ', name='+name +')'
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper


def fortranlike(f, snapshot, mv=False):
    """
    Decorator function for fortran likelihoods
    ==========================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib.
    
    Assume args = (x, param1, param2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the parameters have the same shape as x.

    mv: multivariate (True/False)
    
    Add compatibility with GoF (Goodness of Fit) tests 
    --------------------------------------------------
    * Add a 'prior' keyword (True/False)
    * If the keyword gof is given and is True, return the GoF (Goodness of Fit)
    points instead of the likelihood. 
    * A 'loss' keyword can be given, to specify the loss function used in the 
    computation of the GoF points. 
    * If the keyword random is given and True, return a random variate instead
    of the likelihood.
    """
    name = f.__name__[:-5]
    # Take a snapshot of the main namespace.
    
    
    # Find the functions needed to compute the gof points.
    expval_func = snapshot[name+'_expval']
    random_func = snapshot['r'+name]
    
    def wrapper(*args, **kwds):
        """This wraps a likelihood."""
        
        # Shape manipulations
        if not mv:
            xshape = np.shape(node_to_NDarray(args[0]))
            newargs = [np.asarray(node_to_NDarray(args[0]))]
            for arg in args[1:]:
                newargs.append(np.resize(node_to_NDarray(arg), xshape))
            for key in kwds.iterkeys():
                kwds[key] = node_to_NDarray(kwds[key])  
        else:
            """x, mu, Tau
            x: (kxN)
            mu: (kxN) or (kx1)
            Tau: (k,k)
            """
            xshape=np.shape(args[0])
            newargs = [np.asarray(args[0])] 
            newargs.append(np.resize(args[1], xshape))
            newargs.append(np.asarray(args[2]))
            
        if kwds.pop('gof', False) and not kwds.pop('prior', False):
            """Return gof points."""            
            loss = kwds.pop('gof', squared_loss)
            #name = kwds.pop('name', name)
            expval = expval_func(*newargs[1:], **kwds)
            y = random_func(*newargs[1:], **kwds)
            gof_points = GOFpoints(newargs[0],y,expval,loss)
            return gof_points
        elif kwds.pop('random', False):
            return random_func(*newargs[1:], **kwds)
        else:
            """Return likelihood."""
            try:
                return f(*newargs, **kwds)
            except LikelihoodError:
                return -np.Inf
        

    # Assign function attributes to wrapper.
    wrapper.__doc__ = f.__doc__
    wrapper._PyMC = True
    wrapper.__name__ = f.__name__
    wrapper.name = name
    return wrapper



def priorwrap(f):
    """
    Decorator to create prior functions
    ===================================
    
    Given a likelihood function, return a prior function. 
    
    The only thing that changes is that the _prior attribute is set to True.
    """
    def wrapper(*args, **kwds):
        kwds['prior'] = True
        return f(args, kwds)
    wrapper.__doc__ = string.capwords(f.__name__) + ' prior'
    return wrapper

def magic_set(obj, func, name=None):
    """
    Adds a function/method to an object.  Uses the name of the first
    argument as a hint about whether it is a method (``self``), class
    method (``cls`` or ``klass``), or static method (anything else).
    Works on both instances and classes.

        >>> class color:
        ...     def __init__(self, r, g, b):
        ...         self.r, self.g, self.b = r, g, b
        >>> c = color(0, 1, 0)
        >>> c      # doctest: +ELLIPSIS
        <__main__.color instance at ...>
        >>> @magic_set(color)
        ... def __repr__(self):
        ...     return '<color %s %s %s>' % (self.r, self.g, self.b)
        >>> c
        <color 0 1 0>
        >>> @magic_set(color)
        ... def red(cls):
        ...     return cls(1, 0, 0)
        >>> color.red()
        <color 1 0 0>
        >>> c.red()
        <color 1 0 0>
        >>> @magic_set(color)
        ... def name():
        ...     return 'color'
        >>> color.name()
        'color'
        >>> @magic_set(c)
        ... def name(self):
        ...     return 'red'
        >>> c.name()
        'red'
        >>> @magic_set(c)
        ... def name(cls):
        ...     return cls.__name__
        >>> c.name()
        'color'
        >>> @magic_set(c)
        ... def pr(obj):
        ...     print obj
        >>> c.pr(1)
        1
    """
    
    is_class = (isinstance(obj, type)
                or isinstance(obj, types.ClassType))
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if not args or args[0] not in ('self', 'cls', 'klass'):
        # Static function/method
        if is_class:
            replacement = staticmethod(func)
        else:
            replacement = func
    elif args[0] == 'self':
        if is_class:
            replacement = func
        else:
            def replacement(*args, **kw):
                return func(obj, *args, **kw)
            try:
                replacement.func_name = func.func_name
            except:
                pass
    else:
        if is_class:
            replacement = classmethod(func)
        else:
            def replacement(*args, **kw):
                return func(obj.__class__, *args, **kw)
            try:
                replacement.func_name = func.func_name
            except:
                pass
    if name is None:
        name = func.func_name
    setattr(obj, name, replacement)

# Local dictionary of the likelihood objects
# The presence of objects must be confirmed in __likelihoods__, defined at the
# top of the file.
snapshot = locals().copy()
likelihoods = {}
for name, obj in snapshot.iteritems():
    if name[-5:] == '_like' and name[:-5] in availabledistributions:
        likelihoods[name[:-5]] = snapshot[name]

def add_decorated_likelihoods(obj):
    """Decorate the likelihoods present in the local namespace and
    assign them as methods to obj."""
    for name, like in likelihoods.iteritems():
        magic_set(obj, fortranlike_method(like, snapshot),name+'_like_dec')
        magic_set(obj, priorwrap(fortranlike_method(like, snapshot)),name+'_prior_dec')
#        setattr(obj, name+'_like_dec', classmethod(fortranlike_method(like, obj, snapshot)))
#        setattr(obj, name+'_prior_dec', classmethod(priorwrap(fortranlike_method(like, obj, snapshot))))
            
def local_decorated_likelihoods(obj):
    """New interface likelihoods""" 
    for name, like in likelihoods.iteritems():
        obj[name+'_like'] = fortranlike(like, snapshot)

if __name__=='__main__':
    import __main__
    local_decorated_likelihoods(__main__.__dict__)
else:
    local_decorated_likelihoods(locals())
