# Proposition 3

# Here is the interface I'm proposing for PyMC. The general idea is to create 
# objects that return their likelihood. The dependencies are stored in an 
# attribute named 'parents'. Once the user has defined his parameters, data and 
# nodes, he instantiates a sampler class using those elements. 
#
# See example below.

import numpy as np
from inspect import getargs
import types, copy

class MCArray(np.ndarray):
    def __new__(subtype, name, data, like, class_ref, MCType, info=None, dtype=None, copy=True):
        """data: array
        like: likelihood
        MCType: Data, Parameter.
        """
        subtype._info = info
        subtype.__like = like
        subtype._cref = class_ref
        subtype._name  = name[:]
        subtype._type = MCType[:]
        return np.array(data).view(subtype)

    def __array_finalize__(self,obj):
        self.info = self._info
        self.name = self._name
        self.type = self._type
        self.__repr__ = self.make_repr()
        self._like = self.__like
    
    def like(cself):
            kw = cself._cref.get_args(cself.name)
            return cself._like( **kw)
        
    def make_repr(self):
        if self._type == 'Parameter':
            def repr(self):
                desc="""PyMC Parameter( %(data)s,
              tag=%(tag)s)"""
                return desc % {'data': str(self), 'tag':self.info }
        elif self._type == 'Data':
            def repr(self):
                desc="""PyMC Data( %(data)s,
              tag=%(tag)s)"""
                return desc % {'data': str(self), 'tag':self.info }
        elif self._type == 'Node':
            def repr(self):
                desc="""PyMC Node( %(data)s,
              tag=%(tag)s)"""
                return desc % {'data': str(self), 'tag':self.info }
        return repr
        
    

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
        self.shape = np.shape(self.init_val)
            
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
        self.shape = np.shape(self.value)
        
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
        for obj in args:
            self.__parse_objects([obj.__name__])
        self.__get_args()
        self.__find_children()
        self.__find_types()
        
        # Create attributes from these objects and fill the attributes 
        # dictionary.
        self.attributes = {}
        self.likelihoods = {}
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
        self.children = {}
        for p in self.parent_dic.keys():
            self.children[p] = set()
        for child, parents in self.parent_dic.iteritems():
            for p in parents:
                self.children[p].add(child)
        self.ext_children = {}
        for k,v in self.children.iteritems():
            self.ext_children[k] = v.copy()
            
        for child in self.children.keys():
            self.__find_ext_children(child)
        
    def __find_ext_children(self, name):
        children = self.ext_children[name].copy()
        
        if len(children) != 0:
            for child in children:
                if not self.ext_children[name].issuperset(self.children[child]):
                    self.ext_children[name].update(self.children[child])
                    self.__find_ext_children(name)
                
    def __find_types(self):
        self.parameters = set()
        self.nodes = set()
        self.data = set()
        self.logicals = set()
        for name, obj in self.object_dic.iteritems():
            try:
                if obj.type=='Parameter':
                    self.parameters.add(name)
                elif obj.type =='Node':
                    self.nodes.add(name)
                elif obj.type == 'Data':
                    self.data.add(name)
            except AttributeError:
                self.logicals.add(name)
        
        
    def __get_args(self):
        self.call_args = {}
        self.call_attr = {}

        for name, obj in self.object_dic.iteritems():
            # Case 
            try:
                self.call_args[name] = obj.args[:]
            except:
                self.call_args[name] = self.parent_dic[name][:]
            self.call_attr[name] = \
                dict(zip(self.call_args[name][:],self.call_args[name][:]))
            if self.call_attr[name].has_key('self'):
                self.call_attr[name]['self'] = name
        
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
        
    def likelihood(self, name):
        likes = {}
        if type(name) == str:
            name = [name]
        for n in name:
            likes[n] = self.likelihoods[n].fget(self)
        return likes
    
    def create_attributes(self, name, obj):
        """Create attributes from the object.
        The first attribute, name, returns its own value. 
        If the object is a Data, Parameter or a Node, a second attribute is 
        created, name_like, and it returns the object's likelihood.
        
        Name: name of attribute.
        obj:  reference to the object.
        
        TODO: implement time stamps and cacheing.
        """
        def lget(self):
            """Return likelihood."""
            kwds = self.get_args(name)
            return obj(**kwds)
        attribute = property(lget, doc=obj.__doc__)
        setattr(self.__class__, name+'_like', attribute)
        self.likelihoods[name] = getattr(self.__class__, name+'_like')
        like = getattr(self.__class__, name+'_like')
        try:
            if obj.type == 'Data':            
                #setattr(self, '__'+name, np.asarray(obj.value))
                setattr(self, '__'+name, MCArray(name, obj.value, obj.__call__, self, 'Data', obj.__doc__))
                setattr(getattr(self, '__'+name), 'alike', like)
                def fget(self):
                    """Return value of data."""
                    return getattr(self, '__'+name)
                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
                   
            elif obj.type == 'Parameter':
                
                #setattr(self, '__'+name, np.asarray(obj.init_val))
                setattr(self, '__'+name, MCArray(name, obj.init_val, obj.__call__, self, 'Parameter', obj.__doc__))
                def fget(self):
                    """Return value of parameter.""" 
                    return getattr(self, '__'+name)
    
                def fset(self, value):
                    """Set value of parameter.""" 
                    ar = getattr(self, '__'+name)
                    ar.itemset(value)
                attribute = property(fget, fset, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
                
            elif obj.type == 'Node':
                setattr(self, '__'+name, MCArray(name, np.empty(obj.shape), obj.__call__, self, 'Parameter', obj.__doc__))
                def fget(self):
                    selfname = obj.self.__name__
                    value = self.attributes[selfname].fget(self)
                    o = getattr(self, '__'+name)
                    #o.itemset(value)
                    o[:] = value
                    return o

                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
            
            else:
                raise('Object type not recognized.')
                

        
        except AttributeError:
            if obj.__class__ is types.FunctionType:
                def fget(self):
                    args = self.get_args(name)
                    return obj(**args)
                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
        
        self.attributes[name]=getattr(self.__class__, name)
        


class SamplingMethod(object):
    def __init__(self, model, parameters):
        self.model = model
        if type(parameters) == str:        
            self.parameters = [parameters]            
        else:
            self.parameters = parameters
        
        self.__find_probabilistic_children()
        
    def step(self):
        pass
        
    def tune(self):
        pass
        
    def __find_probabilistic_children(self):
        self.children = set()
        for p in self.parameters:
            self.children.update(self.model.ext_children[p])
        
        self.children -= self.model.logicals
        self.children -= self.model.data
        
    def _get_likelihood(self):
        return sum(self.model.likelihood(self.children).values())
            
    likelihood = property(fget = _get_likelihood)
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
@Node(self=model, shape=input.shape)
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
#print 'alpha_like: ', bunch.alpha_like
print 'alpha.like(): ', bunch.alpha.like()
print 'beta: ', bunch.beta
#print 'beta_like: ', bunch.beta_like
print 'beta.like(): ', bunch.beta.like()
print 'exp_output: ', bunch.exp_output
print 'sim_output: ', bunch.sim_output
#print 'sim_output_like: ', bunch.sim_output_like
print 'sim_output.like(): ', bunch.sim_output.like()
print 'residuals: ', bunch.residuals


S = SamplingMethod(bunch, 'alpha')


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
