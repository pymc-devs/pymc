import sys, inspect
from imp import load_dynamic
import distributions
from PyMC2.PyMCObjects import Parameter, Node, PyMCBase
from copy import copy

#
# Find PyMC object's random children.
#
def extend_children(pymc_object):
    new_children = copy(pymc_object.children)
    need_recursion = False
    node_children = set()
    for child in pymc_object.children:
        if isinstance(child,Node):
            new_children |= child.children
            node_children.add(child)
            need_recursion = True
    pymc_object.children = new_children - node_children
    if need_recursion:
        extend_children(pymc_object)
    return


def _extract(__func__, kwds, keys): 
    """
    Used by decorators parameter and node to inspect declarations
    """
    kwds.update({'doc':__func__.__doc__, 'name':__func__.__name__})
    parents = {}

    def probeFunc(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            kwds.update(dict((k,locals.get(k)) for k in keys))
            sys.settrace(None)
        return probeFunc

    # Get the __func__tions logp and random (complete interface).
    sys.settrace(probeFunc)
    try:
        __func__()
    except:
        if 'logp' in keys:  
            kwds['logp']=__func__
        else:
            kwds['eval'] =__func__

    for key in keys:
        if not kwds.has_key(key):
            kwds[key] = None            
            
    for key in ['logp', 'eval']:
        if key in keys:
            if kwds[key] is None:
                kwds[key] = __func__

    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(__func__)
    try:
        parents.update(dict(zip(args[-len(defaults):], defaults)))

    # No parents at all     
    except TypeError: 
        pass
        
    if parents.has_key('value'):
        value = parents.pop('value')
    else:
        value = None
        
    return (value, parents)

def parameter(__func__=None, **kwds):
    """
    Decorator function for instantiating parameters. Usages:
    
    Medium:
    
        @parameter
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
        
        @parameter(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            return foo(value, parent_name, ...)
            
    Long:

        @parameter
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    
        @parameter(trace=trace_object)
        def A(value = ., parent_name = .,  ...):
            
            def logp(value, parent_name, ...):
                return foo(value, parent_name, ...)
                
            def random(parent_name, ...):
                return bar(parent_name, ...)
                
    where foo() computes the log-probability of the parameter A
    conditional on its value and its parents' values, and bar()
    generates a random value from A's distribution conditional on
    its parents' values.
    """

    def instantiate_p(__func__):
        value, parents = _extract(__func__, kwds, keys)
        if not kwds.has_key('isdata'):
            kwds['isdata'] = False
        if kwds['isdata'] == None:
            kwds['isdata'] = False
        if kwds['trace'] == None:
            kwds['trace'] = True
        if kwds['isdata'] == True:
            kwds['trace'] = False
        kwds['children'] = set()
        return Parameter(value=value, parents=parents, **kwds)      
    keys = ['logp','random','trace','rseed']

    if __func__ is None:
        return instantiate_p
    else:
        instantiate_p.kwds = kwds
        return instantiate_p(__func__)

    return instantiate_p


def node(__func__ = None, **kwds):
    """
    Decorator function instantiating nodes. Usage:
    
    @node
    def B(parent_name = ., ...)
        return baz(parent_name, ...)
        
    @node(trace = trace_object)
    def B(parent_name = ., ...)
        return baz(parent_name, ...)        
        
    where baz returns the node B's value conditional
    on its parents.
    """

    def instantiate_n(__func__):
        junk, parents = _extract(__func__, kwds, keys)
        kwds['children'] = set()
        if kwds['trace'] == None:
            kwds['trace'] = True        
        return Node(parents=parents, **kwds)        
    keys = ['eval','trace']
    
    if __func__ is None:
        return instantiate_n
    else:
        instantiate_n.kwds = kwds
        return instantiate_n(__func__)

    return instantiate_n


def data(__func__=None, **kwds):
    """
    Decorator instantiating data objects. Usage is just like
    parameter.
    """
    return parameter(__func__, isdata=True, trace = False, **kwds)
    
def create_distribution_instantiator(name, logp=None, random=None):
    """Return a function to instantiate a parameter from a particular distribution.
     
      :Example:
        >>> Exponential = create_distribution_instantiator('exponential')
        >>> A = Exponential(name ='A', beta=4)
    """
    

    if logp is None:
        try:
           logp = getattr(distributions, name+"_like")
        except:
            raise "No likelihood found with this name ", name+"_like"
    if random is None:
        try: 
            random = getattr(distributions, 'r'+name)
        except:
            raise "No random generator found with this name ", 'r'+name
        
    
    # Build parents dictionary by parsing the __func__tion's arguments.
    (args, varargs, varkw, defaults) = inspect.getargspec(logp)
    parent_names = args[1:]
    try:
        parents_default = dict(zip(args[-len(defaults):], defaults))
    except TypeError: # No parents at all.   
        parents_default = {}
        
        
    def instantiator(name, trace=True, rseed=False, **kwds):
        # Deal with keywords
        # Find which are parents
        value = kwds.pop('value')
        parents=parents_default
        for k in kwds.keys():
            if k in parent_names:
                parents[k] = kwds.pop(k)
        return Parameter(value=value, name=name, parents=parents, logp=logp, random=random, \
        trace=trace, rseed=rseed, isdata=False, children=set())

    instantiator.__doc__="Instantiate a Parameter instance with a %s prior."%name
    return instantiator
