# Proposition 3

# Here is the interface I'm proposing for PyMC. The general idea is to create 
# objects that return their likelihood. The dependencies are stored in an 
# attribute named 'parents'. Once the user has defined his parameters, data and 
# nodes, he instantiates a sampler class using those elements. 
#
# See example below.

import numpy as np
from inspect import getargs
import types

class MCArray(np.ndarray):
    def __new__(subtype, data, class_ref, like, MCType, info=None, dtype=None, copy=True):
        """data: array
        like: likelihood
        MCType: Data or Parameter.
        """
        subtype._info = info
        subtype.info = subtype._info
        subtype._like = like
        subtype._cref = class_ref
        return np.array(data).view(subtype)

    def __array_finalize__(self,obj):
        if hasattr(obj, "info"):
            # The object already has an info tag: just use it
            self.info = obj.info
            self.like = obj._like
        else:
            # The object has no info tag: use the default
            self.info = self._info
    
    def like(self, **kwargs):
       return self._like(self._cref, **kwargs)

    def __repr__(self):
        desc="""Parameter array( %(data)s,
      tag=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.info }

class Parameter:
    """Decorator class for PyMC parameter.
    
    Input
        func: function returning the likelihood of the parameter, ie its prior. 
        init_val: Initial value to start the sampling.
    
    Example
        @parameter(init_val=56)
        def alpha(self):
            "Parameter alpha of model M."
            return uniform_like(self, 0, 10)
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self.type = 'Parameter'
            
    def __call__(self, func):
        self.args = getargs(func.func_code)[0]
        self.parents = getargs(func.func_code)[0]
        if 'self' in self.parents:
            self.parents.remove('self')
            
                    
        def wrapper(*args, **kwds):
            return func(*args, **kwds)
        wrapper.__dict__.update(self.__dict__)
        wrapper.__doc__ = func.__doc__
        return wrapper
    
class Data(Parameter):
    """Decorator class for PyMC data.
    
    Input
        'value': The data.
        func: A function returning the likelihood of the data.
        
    Example
        @Data(value=[1,2,3,4])
        def input():
            "Input data to force model."
            return 0
    """
 
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self.type = 'Data'
        
class Node(Parameter):
    """Decorator class for PyMC node.

    Input
        self: A function returning the likelihood of the node.
        
    Example
        @Node(self = func)
        def simulation(self, exp_output, var):
            return normal_like(self, exp_output, 1./var)
    
    Note
        All arguments to the likelihood of a Node must be somewhere in the 
        namespace, in order for Node to find the parents (dependencies).
        In the example, sim_output must be a function, and var a constant or a 
        Parameter. 
    
    """
    def __init__(cself, **kwds):
        cself.__dict__.update(kwds)
        cself.type = 'Node'
        

    def __call__(self, func):
        self.args = getargs(func.func_code)[0]
        self.parents = getargs(func.func_code)[0]
        if 'self' in self.parents:
            self.parents.remove('self')
            self.parents.append(self.self.__name__)
                    
        def wrapper(*args, **kwds):
            return func(*args, **kwds)
        wrapper.__dict__.update(self.__dict__)
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        return wrapper

class Bunch(object):
    """Instantiation: Bunch(top object)
    
    For each object and parents of object, create an attribute. 
    
    There are four kinds of attributes, 
    Data: Return its own value. To get its likelihood, call data.like().
            These attributes cannot be set.
    Parameter: Return its own value. To get its likelihood, call 
            parameter.like(). These attributes can be set by parameter = value.
    Node: Return its own likelihood, dependent on its parents value. 
            Cannot be set.
    Function: Return its own value, dependent on its parents value.
            Cannot be set. 
    
    Method
    ------
    likelihood(): Return the global likelihood.
    
    """
    def __init__(self, *args, **kwds):
        # Get all parent objects        
        # Create a dictionnary linking each object to its parents.        
        self.object_dic = {}
        self.parent_dic = {}
        self.call_args = {}
        self.call_attr = {}
        for obj in args:
            self.__parse_objects([obj.__name__])
        self.__get_args()
        self.__find_children()
        
        # Create attributes from these objects and fill the attributes 
        # dictionary.
        self.attributes = {}
        for k,o in self.object_dic.iteritems():        
            self.create_attributes(k,o)
            
        
        # All objects are attributed a value and a like. 
        # underlying the fget method of those attribute is the caching mechanism.
        # All objects are linked, so that the value of parents is known.
        
            
    def __parse_objects(self, obj_name):
        """Get the parents of obj_name from the global namespace."""
        for name in obj_name:
            if name is not None:
                self.object_dic[name]=globals()[name]
                try:
                    # Object is a Data, Parameter or Node instance.
                    parent_names = self.object_dic[name].parents[:]
                except AttributeError:
                    # Object is a plain function.
                    parent_names = getargs(self.object_dic[name].func_code)[0]

                self.parent_dic[name]=parent_names[:]
                self.__parse_objects(parent_names)

    def __find_children(self):
        pass
        
    def __get_args(self):
        for name, obj in self.object_dic.iteritems():
            try:
                self.call_args[name] = obj.args[:]
            except:
                self.call_args[name] = self.parent_dic[name][:]
            self.call_attr[name] = \
                dict(zip(self.call_args[name][:],self.call_args[name][:]))
            try:
                self.call_attr[name]['self'] = name
            except:
                pass
        
    def get_parents(self, attr_name):
        """Return a dictionary with the attribute's parents and their
        current values."""
        parents = self.parent_dic[attr_name][:]
        return dict([(p, self.attributes[p].fget(self)) for p in parents])
        
    def get_args(self, attr_name):
        """Return a dictionary with the attribute's calling arguments and their 
        current value."""
        conversion = self.call_attr[attr_name]
        return dict([(call, self.attributes[a].fget(self)) for call, a in conversion.iteritems()])
        
        
    def create_attributes(self, name, obj):
        """Create attributes from the object.
        The first attribute, name, returns its own value. 
        If the object is a Data, Parameter or a Node, a second attribute is 
        created, name_like, and it returns the object's likelihood.
        
        Name: name of attribute.
        obj:  reference to the object.
        
        TODO: implement time stamps and cacheing.
        """
        
        try:
            if obj.type == 'Data':            
                setattr(self, '__'+name, np.asarray(obj.value))
                def fget(self):
                    """Return value of data."""
                    return getattr(self, '__'+name)
                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
                   
            elif obj.type == 'Parameter':
                setattr(self, '__'+name, np.asarray(obj.init_val))
                
                def fget(self):
                    """Return value of parameter.""" 
                    return getattr(self, '__'+name)
    
                def fset(self, value):
                    """Set value of parameter.""" 
                    setattr(self, '__'+name, value)
                attribute = property(fget, fset, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
                
            elif obj.type == 'Node':
                def fget(self):
                    selfname = obj.self.__name__
                    return self.attributes[selfname].fget(self)

                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
            
            else:
                raise('Object type not recognized.')
                
            def lget(self):
                """Return likelihood."""
                kwds = self.get_args(name)
                return obj(**kwds)
            attribute = property(lget, doc=obj.__doc__)
            setattr(self.__class__, name+'_like', attribute)
        
        except AttributeError:
            if obj.__class__ is types.FunctionType:
                def fget(self):
                    parents = self.get_parents(name)
                    return obj(**parents)
                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
        
        self.attributes[name]=getattr(self.__class__, name)
        

# Example ------------------------------------------------------------------
from test_decorator import normal_like, uniform_like

# Define model parameters
@Parameter(init_val = 4)
def alpha(self):
    """Parameter alpha of toy model."""
    # The return value is the prior. 
    return uniform_like(self, 0, 10)

@Parameter(init_val=5)
def beta(self, alpha):
    """Parameter beta of toy model."""
    return normal_like(self, alpha, 2)


# Define the data
@Data(value = [1,2,3,4])
def input():
    """Measured input driving toy model."""
    like = 0
    return like
    
@Data(value = [45,34,34,65])
def exp_output():
    """Experimental output."""
    # likelihood a value or a function
    return 0
    
# Model function
# No decorator is needed, its just a function.
def model(alpha, beta, input):
    """Return the simulated output.
    Usage: sim_output(alpha, beta, input)
    """
    self = alpha + beta * input
    return self
    

# The likelihood node. 
# The keyword specifies the function to call to get the node's value.
@Node(self=model)
def sim_output(self, exp_output):
    """Return likelihood of simulation given the experimental data."""
    return normal_like(self, exp_output, 2)

# Just a function we want to compute along the way.
def residuals(sim_output, exp_output):
    """Model's residuals"""
    return sim_output-exp_output

# Put everything together
bunch = Bunch(sim_output, residuals)
print 'alpha: ', bunch.alpha
print 'alpha_like: ', bunch.alpha_like
print 'beta: ', bunch.beta
print 'beta_like: ', bunch.beta_like
print 'exp_output: ', bunch.exp_output
print 'sim_output: ', bunch.sim_output
print 'sim_output_like: ', bunch.sim_output_like
print 'residuals: ', bunch.residuals


#print bunch.parent_dic
#print bunch.call_args
# The last step would be to call 
# Sampler(posterior, 'Metropolis')
# i.e. sample the parameters from posterior using a Metropolis algorithm.
# Sampler recursively looks at the parents of posterior, namely sim_output and
# exp_output, identifies the Parameters and sample over them according to the
# posterior likelihood. 
# Since the parents are known for each element, we can find the children of the 
# Parameters, so when one Parameter is sampled, we only need to compute the 
# likelihood of its children, and avoid computing elements that are not modified 
# by the current Parameter. 
