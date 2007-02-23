from numpy import array, zeros, ones, arange
from AbstractBase import *

# These are slower than the C versions by:
# 1.3X for test_MCMC
# 3.7X for test_fast
# 2.2X for test_joint
# Not too bad.
# The bulk of the time tends to be spent in _check_for_recompute.
# I've switched all the caches, etc to ndarrays, so it should be possible
# to write those methods in Weave.

# We ought to be able to use weave on the following code, but I haven't tried it.
# node._check_for_recompute = code_segs[0] + code_segs[2] + code_segs[4]
# param._check_for_recompute = code_segs[0] + code_segs[1] + code_segs[2] + code_segs[3] + code_segs[4]

code_segs = [ """\\Segment 0

//Return value will be index.
int i, j, index, mismatch;
for(j=0;j<self._cache_depth;j++){
    mismatch=0;         

""",
"""\\Segment 1

    // Check self's timestamp (parameter only)
    if(self.timestamp != self._timestamp_caches(j)) mismatch=1;

""",
"""\\Segment 2

    // Check parents' timestamps (parameter and node)
    if(mismatch==0){
        for(i=0;i<self.N_node_parents;i++){
            if(self._node_parent_timestamps(i)!=self._node_parent_timestamp_caches(j,i)){
                mismatch=1;
                break;
            }
        }
    }
    
    if(mismatch==0){
        for(i=0;i<self.N_param_parents;i++){
            if(self._param_parent_timestamps(i)!=self._param_parent_timestamp_caches(j,i)){
                mismatch=1;
                break;
            }           
        }
    }           
    
    // If control reaches here, the value is cached.
    if(mismatch==0) index = j;
}

// If control reaches here, the value is not cached.
""",
"""\\Segment 3

// Preemptively cache self's timestamp (parameter only)
for(j=1;j<self._cache_depth;j++){
    self._timestamp_caches(j) = self._timestamp_caches(j-1)
}
self._timestamp_caches(0) = self.timestamp

""",
"""\\Segment 4

// Preemptively cache parents' timestamps (parameter and node)
for(i=0;i<self.N_node_parents;i++){
    for(j=1;j<self._cache_depth;j++){
        self._node_parent_timestamp_caches(j,i) = self._node_parent_timestamp_caches(j-1,i)
    }
    self._node_parent_timestamp_caches(0,i) = self._node_parent_timestamps(i);
}

for(i=0;i<self.N_param_parents;i++){
    for(j=1;j<self._cache_depth;j++){
        self._param_parent_timestamp_caches(j,i) = self._param_parent_timestamp_caches(j-1,i)
    }
    self._param_parent_timestamp_caches(0,i) = self._param_parent_timestamps(i);
}

index -1;
"""]


class PurePyMCObject(PurePyMCBase):
    def __init__(self, doc, name, parents, cache_depth, trace):

        self.parents = parents
        self.children = set()
        self.__doc__ = doc
        self.__name__ = name
        self._value = None
        self.trace = trace

        self._cache_depth = cache_depth
        
        # Some aranges ahead of time for faster looping
        self._cache_range = arange(self._cache_depth)
        self._upper_cache_range = arange(1,self._cache_depth)
        
        # Timestamp is an `array scalar' so it stays put in memory.
        # That means self._node_parent_timestamps and 
        # self._param_parent_timestamps can contain references
        # directly to the parents' timestamp objects and skip the
        # attribute lookup.     
        self.timestamp = array(0,dtype=int)     

        # Find self's parents that are pymc objects,
        # prepare caches,
        # and add self to pymc object parents' children sets
        self.file_parents()

    def file_parents(self):
        
        # A dictionary of those parents that are PyMC objects or containers.
        self._pymc_object_parents = {}

        # The parent_values dictionary will get passed to the logp/
        # eval function.        
        self.parent_values = {}
        
        self.N_node_parents = 0
        self.N_param_parents = 0
                
        # Make sure no parents are None, and count up the parents
        # that are parameters and nodes, including those enclosed
        # in PyMC object containers.
        for key in self.parents.iterkeys():
            assert self.parents[key] is not None, self.__name__ + ': Error, parent ' + key + ' is None.'
            if isinstance(self.parents[key], ParameterBase):
                self.N_param_parents += 1
            elif isinstance(self.parents[key], NodeBase):
                self.N_node_parents += 1
            elif isinstance(self.parents[key], ContainerBase):
                self.N_node_parents += len(self.parents[key].nodes)
                self.N_param_parents += len(self.parents[key].parameters)
        
        # More upfront aranges for faster looping.
        self._node_range = arange(self.N_node_parents)
        self._param_range = arange(self.N_param_parents)                
        
        # Initialize array of references to parents' timestamps.
        self._node_parent_timestamps = zeros(self.N_node_parents,dtype=object)
        self._param_parent_timestamps = zeros(self.N_param_parents,dtype=object)
        
        # Initialize parent timestamp cache arrays
        self._node_parent_timestamp_caches = -1*ones((self._cache_depth, self.N_node_parents), dtype=int)
        self._param_parent_timestamp_caches = -1*ones((self._cache_depth, self.N_param_parents), dtype=int)


        # Sync up parents and children, figure out which parents are PyMC
        # objects and which are just objects.
        #
        # ultimate_index indexes the parents, including those enclosed in
        # containers.
        ultimate_index=0
        for key in self.parents.iterkeys():
            
            if isinstance(self.parents[key],PyMCBase):

                # Add self to this parent's children set
                self.parents[key].children.add(self)

                # Remember that this parent is a PyMCBase.
                # This speeds the _refresh_parent_values method.
                self._pymc_object_parents[key] = self.parents[key]
                
                # Record references to the parent's timestamp array scalars
                if isinstance(self.parents[key],NodeBase):
                    self._node_parent_timestamps[ultimate_index] = self.parents[key].timestamp

                if isinstance(self.parents[key],ParameterBase):
                    self._param_parent_timestamps[ultimate_index] = self.parents[key].timestamp
                    
                ultimate_index += 1                 
            
            # Unpack parameters and nodes from containers 
            # for timestamp=checking purposes.
            elif isinstance(self.parents[key], ContainerBase):          

                # Record references to the parent's parameters' 
                # and nodes' timestamp array scalars
                for node in self.parents[key].nodes:
                    self._node_parent_timestamps[ultimate_index] = node.timestamp
                    ultimate_index += 1

                for param in self.parents[key].paramters:
                    self._param_parent_timestamps[ultimate_index] = param.timestamp
                    ultimate_index += 1                 

            # If the parent isn't a PyMC object or PyMC object container,
            # record a reference to its value.
            else:
                self.parent_values[key] = self.parents[key]
    
    # Extract the values of parents that are PyMC objects or containers.
    # Don't worry about unpacking the containers, see their value attribute.
    def _refresh_parent_values(self):
        for item in self._pymc_object_parents.iteritems():
            self.parent_values[item[0]] = item[1].value



        
class PureNode(PurePyMCObject,NodeBase):

    def __init__(self, eval,  doc, name, parents, trace=True, cache_depth=2):

        PurePyMCObject.__init__(self, 
                                doc=doc, 
                                name=name, 
                                parents=parents, 
                                cache_depth = cache_depth, 
                                trace=trace)

        # This function gets used to evaluate self's value.
        self._eval_fun = eval

        # Caches of recent computations of self's value
        self._cached_value = []
        for i in range(self._cache_depth): 
            self._cached_value.append(None)

    #
    # Define the attribute value.
    #
    
    # See if a recompute is necessary.
    def _check_for_recompute(self):
        for i in self._cache_range:
            mismatch=False

            for j in self._node_range:
                if not self._node_parent_timestamps[j] == self._node_parent_timestamp_caches[i][j]:
                    mismatch=True
                    break
                    
            if not mismatch:
                for j in self._param_range:
                    if not self._param_parent_timestamps[j] == self._param_parent_timestamp_caches[i][j]:
                        mismatch=True
                        break

            if not mismatch:
                return i
            
        # If control reaches here, a mismatch occurred.
        for j in self._node_range:
            for i in self._upper_cache_range:
                self._node_parent_timestamp_caches[i][j] = self._node_parent_timestamp_caches[i][j-1]
            self._node_parent_timestamp_caches[0][j] = self._node_parent_timestamps[j]
            
        for j in self._param_range:
            for i in self._upper_cache_range:
                self._param_parent_timestamp_caches[i][j] = self._param_parent_timestamp_caches[i][j-1]
            self._param_parent_timestamp_caches[0][j] = self._param_parent_timestamps[j]

        return -1;


    def get_value(self):

        self._refresh_parent_values()
        recomp = self._check_for_recompute()
        
        if recomp < 0:
        
            #Recompute
            self._value = self._eval_fun(**self.parent_values)
        
            # Cache and increment timestamp
            del self._cached_value[self._cache_depth-1]
            self._cached_value.insert(0,self._value)
            self.timestamp += 1
        
        else: self._value = self._cached_value[recomp]
        
        return self._value
        
    def set_value(self,value):
        raise AttributeError, 'Node '+self.__name__+'\'s value cannot be set.'

    value = property(fget = get_value, fset=set_value)
    




class PureParameter(PurePyMCObject, ParameterBase):

    def __init__(   self, 
                    logp, 
                    doc, 
                    name, 
                    parents, 
                    random = None, 
                    trace=True, 
                    value=None, 
                    rseed=False, 
                    isdata=False,
                    cache_depth=2):

        PurePyMCObject.__init__(self, 
                                doc=doc, 
                                name=name, 
                                parents=parents, 
                                cache_depth=cache_depth, 
                                trace=trace)

        # A flag indicating whether self's value has been observed.
        self.isdata = isdata
        
        # This function will be used to evaluate self's log probability.
        self._logp_fun = logp
        
        # This function will be used to draw values for self conditional on self's parents.
        self._random = random
        
        # A seed for self's rng. If provided, the initial value will be drawn. Otherwise it's
        # taken from the constructor.
        self.rseed = rseed
        if self.rseed and self._random:
            self._value = self.random()
        else:
            self._value = value         

        # Caches for recent computations of self's log probability.
        self._cached_logp = []
        for i in range(self._cache_depth):
            self._cached_logp.append(None)

        # Initialize own timestamp cache array
        self._timestamp_caches = -1 * ones(self._cache_depth, dtype=int)        


    #
    # Define value attribute
    #
    
    def get_value(self):
        return self._value

    # Record new value and increment timestamp
    def set_value(self, value):
        
        # Value can't be updated if isdata=True
        if self.isdata:
            raise AttributeError, self.__name__+'\'s Value cannot be updated if isdata flag is set'
            
        # Save current value as last_value
        self.last_value = self._value
        self._value = value
        
        self.timestamp += 1

    value = property(fget=get_value, fset=set_value)


    #
    # Define logp attribute
    #
    
    def _check_for_recompute(self):
        for i in self._cache_range:
            mismatch = not (self._timestamp_caches[i] == self.timestamp)
            
            if not mismatch:
                for j in self._node_range:
                    if not self._node_parent_timestamps[j] == self._node_parent_timestamp_caches[i][j]:
                        mismatch=True
                        break

            if not mismatch:
                for j in self._param_range:
                    if not self._param_parent_timestamps[j] == self._param_parent_timestamp_caches[i][j]:
                        mismatch=True
                        break

            if not mismatch:
                return i
            
        # If control reaches here, a mismatch occurred.
        for j in self._upper_cache_range:
            self._timestamp_caches[j] = self._timestamp_caches[j-1]
        self._timestamp_caches[0] = self.timestamp
        
        for j in self._node_range:
            for i in self._upper_cache_range:
                self._node_parent_timestamp_caches[i][j] = self._node_parent_timestamp_caches[i][j-1]
            self._node_parent_timestamp_caches[0][j] = self._node_parent_timestamps[j]
            
        for j in self._param_range:
            for i in self._upper_cache_range:
                self._param_parent_timestamp_caches[i][j] = self._param_parent_timestamp_caches[i][j-1]
            self._param_parent_timestamp_caches[0][j] = self._param_parent_timestamps[j]

        return -1;
        
    def get_logp(self):
        
        self._refresh_parent_values()
        recomp = self._check_for_recompute()
        if recomp<0:
        
            #Recompute
            self._logp = self._logp_fun(self._value, **self.parent_values)
            del self._cached_logp[self._cache_depth-1]
            self._cached_logp.insert(0,self._logp)
        
        else: self._logp = self._cached_logp[recomp]
        
        return self._logp
    
    def set_logp(self):
        raise AttributeError, 'Parameter '+self.__name__+'\'s logp attribute cannot be set'

    logp = property(fget = get_logp, fset=set_logp)


    #
    # Call this when rejecting a jump.
    #
    
    def revert(self):
        """
        Call this when rejecting a jump.
        """
        self._value = self.last_value
        self.timestamp -= 1

    #
    # Sample self's value conditional on parents.
    #
    
    def random(self):
        """
        Sample self conditional on parents.
        """
        if self._random:
            self.value = self._random(**self.parent_values)
        else:
            raise AttributeError, 'Parameter '+self.__name__+' does not know how to draw its value, see documentation'
        return self._value
