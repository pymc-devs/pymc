#-------------------------------------------------------------
# Decorators
#-------------------------------------------------------------

import numpy as np
from numpy import inf, random, sqrt
import string
import inspect
import types, copy
import distributions
from Node import ZeroProbability

def dtrm_to_NDarray(arg):
    if isinstance(arg,proposition5.Deterministic):
        return arg.value
    else:
        return arg

def fortranlike_method(f, snapshot, mv=False):
    """
    Decorator function building likelihood method for Sampler
    =========================================================
    
    Wrap function f(*args, **kwds) where f is a likelihood defined in flib to
    create a method for the Sampler class. Must be called inside the class.
    
    self: Object the method will be assigned to.
    snapshot: Snapshot of the distributions local dictionary.
    mv: multivariate (True/False)
    
    Assume args = (self, x, stoch1, stoch2, ...)
    Before passing the arguments to the function, the wrapper makes sure that 
    the stochs have the same shape as x.


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
            except ZeroProbability:
                return -np.Inf
        

    # Assign function attributes to wrapper.
##    wrapper.__doc__ = f.__doc__+'\n'+like.__name__+\
##        '(self, '+string.join(f.func_code.co_varnames, ', ')+\
##        ', name='+name +')'
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
                replacement.__doc__ = func.__doc__
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
                replacement.__doc__ = func.__doc__
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


def create_distribution_instantiator(name, logp=None, random=None, module=distributions):
    """Return a function to instantiate a stoch from a particular distribution.
     
      :Example:
        >>> Exponential = create_distribution_instantiator('exponential')
        >>> A = Exponential(beta=4)
    """
    
    if type(module) is types.ModuleType:
        module = copy.copy(module.__dict__)
    elif type(module) is dict:
        module = copy.copy(module)
    else:
        raise AttributeError
        
    if logp is None:
        try:
           logp = module[name+"_like"]
        except:
            raise "No likelihood found with this name ", name+"_like"
    if random is None:
        try: 
            random = module['r'+name]
        except:
            raise "No random generator found with this name ", 'r'+name
        
    
    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.   
        parents_default = {}
        
        
    def instantiator(**kwds):
        # Deal with keywords
        # Find which are parents
        value = kwds.pop('value')
        parents=parents_default
        for k in kwds.keys():
            if k in parents:
                parents[k] = kwds.pop(k)
        return Stochastic(value=value, parents=parent, **kwds)

    instantiator.__doc__="Instantiate a Stochastic instance with a %s prior."%name
    return instantiator
    
    
    # Find the names of the parents.
    inspect.getlogp
if __name__=='__main__':
    import __main__
    local_decorated_likelihoods(__main__.__dict__)
else:
    local_decorated_likelihoods(locals())
