# Proposition 3

# Here is the interface I'm proposing for PyMC. The general idea is to create 
# objects (Data, Parameters, Nodes) that return their likelihood, ie, plain 
# functions with a couple of extra attributes.  The main advantage is that 
# it's easy in this context to reuse functions defined elsewhere. 
#
# Once the user has defined his parameters, data, nodes and any other function 
# (logical), he builds a model by merging those elements: 
# >>> model = BuildModel(nodes, logicals, parameters)
#
# This model is the backbone of the Sampling classes as it provides the 
# interface with the values and likelihoods. With the model, it's easy to 
# change a parameter and see how it affects the value and likelihood of the 
# other elements in the model. 
# For example, assume there is a parameter alpha and a node simulation.
# >>> model.alpha = 2
# >>> print model.alpha.like()
# >>> print model.simulation
# >>> print model.simulation.like()
#
# Once the user has a working model, he defines the sampling methods that will 
# be used, using the different classes that will eventually be there. 
# >>> J = JointSampling(model, ['alpha', 'beta'])
# >>> S = Sampler(J) # Takes care of parameters not in J, by assigning a 
# >>> S.sample()     # default sampling method.
#
# Some things to think about:
# Subclassing NDArray is maybe too much. The other solution is to provide
# an attribute model.alpha_like instead of model.alpha.like(). On the other 
# hand, using MCArray allows us to define quickly other methods (gof(), or 
# random() for instance).

from __future__ import division
import numpy as np
from inspect import getargs
import types, copy,sys
from test_decorator import rnormal
from numpy.random import rand


class LikelihoodError(ValueError):
    "Log-likelihood is invalid or negative informationnite"

def Property(function):
    """Decorator to quickly define class properties."""
    keys = 'fget', 'fset', 'fdel'
    func_locals = {'doc':function.__doc__}
    def probeFunc(frame, event, arg):
        if event == 'return':
            locals = frame.f_locals
            func_locals.update(dict((k,locals.get(k)) for k in keys))
            sys.settrace(None)
        return probeFunc
    sys.settrace(probeFunc)
    function()
    return property(**func_locals)

class MCArray(np.ndarray):
    """A subclass of NDArray to provide the likelihood method."""
    def __new__(subtype, data, name=None, like=None, class_ref=None,\
            MCType=None, info=None, dtype=None, copy=True):
        """data: array
        like: likelihood
        MCType: Data, Parameter.
        """
        subtype._info = info
        subtype.__like = like
        subtype._cref = class_ref
        subtype._name  = name
        subtype._type = MCType or ''
        return np.array(data).view(subtype)

    def __array_finalize__(self,obj):
        self.info = self._info
        self.name = self._name
        self.type = self._type
        self._like = self.__like
    
    def like(cself):
            kw = cself._cref.get_args(cself.name)
            return cself._like( **kw)
        
    def itemset(self, *args):
        "Set Python scalar into array"
        item = args[-1]
        args = args[:-1]
        self[args] = item        
        
    def __repr1__(self):
        desc="MCArray(%(data)s, name=%(name)s, like=%(like)s, class_ref=%(cref)s, MCType=%(type)s, info=%(info)s"
        return desc % {'info':self.info, 'type':self.type, 'data': str(self), \
    'name':self.name, 'cref':self._cref, 'like':self._like}

class Parameter:
    """Decorator class for PyMC parameters.
    
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
        wrapper.__name__ = func.__name__
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
    """Decorator class for PyMC nodes.

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

class BuildModel(object):
    """Class to assemble the different components of a probabilistic model.
    
    This is the base object that the different samplers will access. 
    
    Instantiation: Merge(objects)
    
    For each object and parents of object, create an attribute. 
    
    There are four kinds of attributes, 
    Data: Return its own value. To get its likelihood, call data.like().
            These attributes cannot be set.
    Parameter: Return its own value. To get its likelihood, call 
            parameter.like(). These attributes can be set by parameter = value.
    Node: Return its own likelihood, dependent on its parents value. 
            Cannot be set.
    Logical: Return its own value, dependent on its parents value.
             This is generally a function returning a value we want to keep 
            track of. 
            Cannot be set. 

    Other attributes
    ----------------
    parent: Dictionary mapping attributes to their parents.
    object: Dictionary mapping attributes to their functional object.
    attributes: Dictionary maping names to their property objects.
    parameters: Set of Parameters.
    nodes: Set of Nodes.
    data: Set of Data.
    logicals: Set of logical functions. 
    
    Public methods
    --------------
    likelihood(name): Return the current likelihood of name.
    get_value(name): Return the current value of name. 
    set_value(name, new_value): Set the attribute's value to new_value. 
    
    Each Data, Parameter and Node attribute has a like method returning its 
    own likelihood.
    
    """
    def __init__(self, *args, **kwds):
        # Get all parent objects        
        # Create a dictionnary linking each object to its parents.        
       
        self.__objects = {}
        self.__parents = {}
        import __main__
        self.snapshot = __main__.__dict__

        for obj in args:
            self.__parse_objects([obj.__name__])
        self.__fill_call_map()
        self.__find_children()
        self.__fill_types()
        
        # Create attributes from these objects and fill the __attributes 
        # dictionary.
        self.__attributes = {}
        self.likelihoods = {}
        for k,o in self.__objects.iteritems():        
            self.__create_attributes(k,o)
            
        # All objects are attributed a value and a like. 
        # underlying the fget method of those attribute is the caching mechanism.
        # All objects are linked, so that the value of parents is known.

    def __parse_objects(self, obj_name):
        """Get the parents of obj_name from the global namespace."""
        for name in obj_name:
            if name is not None:
                try:
                    self.__objects[name]=self.snapshot[name]
                    try:
                        # Object is a Data, Parameter or Node instance.
                        parent_names = self.__objects[name].parents[:]
                    except AttributeError:
                        # Object is a plain function.
                        parent_names = getargs(self.__objects[name].func_code)[0]
                except KeyError:
                    raise 'Object %s has not been defined.' % name
                self.__parents[name]=parent_names[:]
                self.__parse_objects(parent_names)

    def __find_children(self):
        self.__children = {}
        for p in self.__parents.keys():
            self.__children[p] = set()
        for child, parent in self.__parents.iteritems():
            for p in parent:
                self.__children[p].add(child)
        self.__ext_children = {}
        for k,v in self.__children.iteritems():
            self.__ext_children[k] = v.copy()
            
        for child in self.__children.keys():
            self.__find_ext_children(child)
        
    def __find_ext_children(self, name):
        children = self.__ext_children[name].copy()
        
        if len(children) != 0:
            for child in children:
                if not self.__ext_children[name].issuperset(self.__children[child]):
                    self.__ext_children[name].update(self.__children[child])
                    self.__find_ext_children(name)
                
    def __fill_types(self):
        """Fill the four sets parameters, nodes, data and logicals."""
        self.__parameters = set()
        self.__nodes = set()
        self.__data = set()
        self.__logicals = set()
        for name, obj in self.__objects.iteritems():
            try:
                if obj.type=='Parameter':
                    self.parameters.add(name)
                elif obj.type =='Node':
                    self.nodes.add(name)
                elif obj.type == 'Data':
                    self.data.add(name)
            except AttributeError:
                self.logicals.add(name)
        
    @Property
    def parameters():
        """Set of Parameters."""
        def fget(self):
            return self.__parameters
        
    @Property
    def data():
        """Set of Data in the model."""
        def fget(self):
            return self.__data
        
    @Property
    def nodes():
        """Set of Nodes in the model."""
        def fget(self):
            return self.__nodes
            
    @Property
    def logicals():
        """Set of logical functions in the model."""
        def fget(self):
            return self.__logicals
        
    def __fill_call_map(self):
        """Fill a dictionary __call_map describing for each function the names of the 
        calling arguments and the corresponding attribute names.
        
        Example:
        @Parameter(init_val=1)
        def alpha(self, tau):
            return normal_like(self, 0, tau)
        
        __call_map['alpha'] = {'self':'alpha', 'tau':'tau'}
        """
        self.__call_map = {}
        
        for name, obj in self.__objects.iteritems():
            # Case 
            try:
                self.__call_map[name] = dict.fromkeys(obj.args[:])
            except:
                self.__call_map[name] = dict.fromkeys(self.__parents[name][:])
            
            for key in self.__call_map[name].keys():
                self.__call_map[name][key] = key
            
            # Set self to one's own name.
            if self.__call_map[name].has_key('self'):
                self.__call_map[name]['self'] = name
        
    @Property
    def parents():
        """Dictionary with the attribute's parents."""
        def fget(self):
            return self.__parents
        
    @Property
    def children():
        """Dictionary with the attribute's children."""
        def fget(self):
            return self.__children
        
    @Property
    def ext_children():
        """Dictionary with the children found recursively."""
        def fget(self):
            return self.__ext_children
        
    @Property
    def objects():
        """Dictionary of the functional objects."""
        def fget(self):
            return self.__objects
           
    @Property
    def attributes():
        """Dictionary of the object properties."""
        def fget(self):
            return self.__attributes
        
    def get_args(self, name):
        """Return a dictionary with the attribute's calling arguments and their 
        current value."""
        conversion = self.__call_map[name]
        return dict([(call_name, self.__attributes[a].fget(self)) for call_name, a in conversion.iteritems()])
        
    def likelihood(self, name=None):
        """Return the likelihood of the attributes.
        
        Return a list if name is a list.
        
        name: name or list of name of the attributes.
            Defaults to None, for which it returns a dictionary of the 
            likelihoods of all Nodes and Parameters.
        """
        
        if type(name) == str:
            return self.likelihoods[name].fget(self)
        
        if np.iterable(name):
            return [self.likelihoods[n].fget(self) for n in name]
            
        if name is None:
            likes = {}
            name = self.parameters | self.nodes
            for n in name:
                likes[n] = self.likelihoods[n].fget(self)
            return likes
        
    
    def get_value(self, name=None):
        """Return the values of the attributes.
        If only one name is given, return the value. 
        If a sequence of name is given, return a dictionary. 
        
        Default return values of all Nodes and Parameters.
        """
        if (type(name) == str):
            return self.__attributes[name].fget(self)
        
        values = {}
        if name is None:
            name = self.parameters | self.nodes 
        for n in name:
            values[n] = self.__attributes[n].fget(self)
        return values
        
    def set_value(self, name, value):
        """Set the value of a Parameter."""
        self.__attributes[name].fset(self, value)
        
    def __create_attributes(self, name, obj):
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
                setattr(self, '__'+name, MCArray(obj.value, name, obj.__call__,\
                 self, 'Data', obj.__doc__))
                setattr(getattr(self, '__'+name), 'alike', like)
                def fget(self):
                    """Return value of data."""
                    return getattr(self, '__'+name)
                attribute = property(fget, doc=obj.__doc__)
                setattr(self.__class__, name, attribute)
                
                   
            elif obj.type == 'Parameter':
                
                #setattr(self, '__'+name, np.asarray(obj.init_val))
                setattr(self, '__'+name, MCArray(obj.init_val, name, \
                obj.__call__, self, 'Parameter', obj.__doc__))
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
                setattr(self, '__'+name, MCArray(np.empty(obj.shape), name, obj.__call__, self, 'Parameter', obj.__doc__))
                def fget(self):
                    selfname = obj.self.__name__
                    value = self.__attributes[selfname].fget(self)
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
                
        
        self.__attributes[name]=getattr(self.__class__, name)
        


class SamplingMethod(object):
    """Basic Metropolis sampling for scalars."""
    def __init__(self, model, parameter, dist=rnormal, debug=False):
        self.model = model
        self.names = parameter
        self._asf = 1
        self.dist = dist
        self.DEBUG = debug
        self.__find_probabilistic_children()
        self.current = copy.copy(self.model.get_value(self.names))
        self.current_like = self._get_likelihood()
        self.accepted = 0
        self.rejected = 0 
        
    def step(self):
        self.sample_candidate()
        accept = self.test()
        
        if self.DEBUG:
            print '%-20s%20s%20s' % (self.names, 'Value', 'Likelihood')
            print '%10s%-10s%20f%20f' % ('', 'Current', self.current, self.current_like)
            print '%10s%-10s%20f%20f' % ('', 'Candidate', self.candidate, self.candidate_like)
            print '%10s%-10s%20s\n' % ('', 'Accepted', str(accept))
        
        if accept:
            self.accept()
        else:
            self.reject()
        
    def sample_candidate(self):
        self.candidate = self.current + self.dist(0, self._asf)
        self.model.set_value(self.names, self.candidate)
        self.candidate_like = self._get_likelihood()

    def tune(self, divergence_threshold=1e10, verbose=False):
        """
        Tunes the scaling hyperparameter for the proposal distribution
        according to the acceptance rate of the last k proposals:
        
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        
        This method is called exclusively during the burn-in period of the
        sampling algorithm.
        """
        # Calculate recent acceptance rate        
        acc_rate = self.accepted/(self.accepted + self.rejected)

        if verbose:
            print
            print 'Tuning', self.name
            print '\tcurrent value:', self.current_value
            print '\tcurrent proposal hyperparameter:', self._asf
        
        
        acc_rate = 1.0 - self._rejected*1.0/int_length
        
        tuning = True
        
        # Switch statement
        if acc_rate<0.001:
            # reduce by 90 percent
            self._asf *= 0.1
        elif acc_rate<0.05:
            # reduce by 50 percent
            self._asf *= 0.5
        elif acc_rate<0.2:
            # reduce by ten percent
            self._asf *= 0.9
        elif acc_rate>0.95:
            # increase by factor of ten
            self._asf *= 10.0
        elif acc_rate>0.75:
            # increase by double
            self._asf *= 2.0
        elif acc_rate>0.5:
            # increase by ten percent
            self._asf *= 1.1
        else:
            tuning = False
        
        # Re-initialize rejection count
        self.accepted = 0
        self.rejected = 0 
        
        # If the scaling factor is diverging, abort
        if self._asf > divergence_threshold:
            raise DivergenceError, 'Proposal distribution variance diverged'
        
        
        if verbose:
            print '\tacceptance rate:', acc_rate
            print '\tadaptive scaling factor:', self._asf
            print '\tnew proposal hyperparameter:', self._hyp*self._asf
        
        return tuning
        
        
        
    def test(self):
        alpha = self.candidate_like - self.current_like
        if alpha > 0 or np.exp(alpha) >= rand():
            return True
        else:
            return False
            
    def accept(self):
        self.current = self.candidate
        self.current_like = self.candidate_like
        self.accepted += 1
        
    def reject(self):
        self.model.set_value(self.names, self.current)
        self.rejected += 1
        
    def __find_probabilistic_children(self):
        self.__children = set()
        if type(self.names) == str:
            params = [self.names]
        else:
            params = self.names
        for p in params:
            self.__children.update(self.model.ext_children[p])
        
        self.__children -= self.model.logicals
        self.__children -= self.model.data
        
    def _get_likelihood(self):
        """Return the joint likelihood of the parameters and their children."""
        try:
            ownlike = self.model.likelihood(self.names)
            childlike = sum(self.model.likelihood(self.__children))
            like = ownlike+childlike
        except (LikelihoodError, OverflowError, ZeroDivisionError):
            like -np.Inf
        return like
        
    @Property
    def likelihood():
        """Likelihood of the parameter and its children.
        The likelihood of parameters and nodes that are independent of this 
        parameter is not included in the compution.
        """
        def fget(self):
            return self._get_likelihood()
        

class JointSampling(SamplingMethod):
    """We need to redefine step, tune and sample_candidate. 
    The rest should be identical."""
    def step(self):
        pass
        
    def tune(self):
        pass
    
    def sample_candidate(self):
        pass
        
class Sampler(object):
    pass
