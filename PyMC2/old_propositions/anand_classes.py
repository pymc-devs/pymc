
class Node(Object):
    
    def __init__(self, name, sampler=None, init_val=None, observed=False, shape=None, plot=True):
        """Class initialization"""
    
        self.name = name
    
        """
        Initialize the state counter, which is incremented every time self.set_value
        is called, and state_counter_of_prior, which is the last value of state_counter
        for which self's prior probability (conditional on self's parents) was computed.
    
        See Parameter.get_prior to see the state_counters in action.
        """
        self.state_counter = -1
        self.state_counter_of_prior = None
    
        #Initialize value
        self.set_value(0.)
        if shape:
                self.set_value(zeros(shape, 'd'))
    
        if init_val is not None:
                self.set_value(init_val)
    
        # Parents is a dictionary. A parent is accessed by its parameter name.
        self.parents = {}
    
        """
        Parent_eval_args is a dictionary, keyed just like parents. It holds the
        arguments that should be passed to parents when querying their state.
        For example, if parent 'mu' is a 100 X 100 matrix but self only depends on
        element (10,43), parent_eval_args['mu'] could be set to (10,43).
        """
        self.parent_eval_args = {}
    
        """
        Parent_state_counters_of_prior is another dictionary keyed just like parents.
        It holds the value of the parents' state_counters that were recorded
        last time self's prior probability was computed.
    
        See Parameter.get_prior to see the state counters in action.
        """
        self.parent_state_counters_of_prior = {}
    
        """
        Children is a set, because each child should be counted only once.
        Since self will never want to refer to its children by name, there's
        no need for children to be keyed in a dictionary like parents are.
        """
        self.children = set([])
    
        """
        Is the value of this node known with no uncertainty?
        That is, does this node hold data?
        """
        self.observed = observed
    
        # Specify sampler, do the flag for plotting, initialize the traces, etc.
    
    
    
    def add_parent(self, new_parent, param_name, eval_argument = None):
    
        """
        This function adds a parent to self's parents dictionary under key
        param_name, and adds eval_argument to self's parent_eval_args dictionary
        under key param_name.
    
        It also adds self to new_parent's children set.
    
        There is no add_child method, because it wouldn't be clear what parent key
        self should pass to the new child.
    
        NOTE: One way we could lighten the implementation up considerably would be
        to find some nice way to pass a new Parameter's parents into its constructor,
        with keys. For example, if C~N(A,B),
    
        B = Parameter('mu'=A, 'V' = B),
    
        but something that would actually work. This would be pretty close to the
        Bayesian statements you mentioned.
    
        Also, it would be nice if a parameter's parents were allowed to be plain
        numpy arrays instead of actual parameters.
        """
    
        # Do I already have a parent under this name? If so, kick it out.
        if self.parents.has_key (param_name):
                if self.parents[param_name] is not None:
                        self.remove_parent(param_name)
    
        # Add the new parent to parents, and the new argument to parent_eval_args.
        self.parents[param_name] = new_parent
        self.parent_eval_args[param_name] = eval_argument
    
        # Add self to the new parent's children set.
        new_parent.children.add(self)
    
    def remove_parent(self, bad_parent_key):
    
        """
        Pretty self-explanatory, but there's a fair bit of bookkeeping involved.
        This method should be useless to the user unless they're building a model
        from an interactive session.
        """
    
        if self.parents[bad_parent_key] is None:
                print "Warning, attempted to remove nonexistent parent ",bad_parent_key," from node ", self.name
    
        else:
                """
                If this parent is included in my parents dictionary under only one key,
                remove me from its children set.
    
                If, on the other hand, this parent is included in my parents dictionary under
                multiple keys, I still depend on it, so don't remove me from its children
                set.
                """
                if self.parents.values().count(parents[bad_parent_key]) == 1:
                        self.parents[bad_parent_key].children.remove(self)
    
                """
                Either way, expunge the desired entries in my parents and
                parent_eval_args dictionaries.
                """
                self.parents[bad_parent_key] = None
                self.parent_eval_args[bad_parent_key] = None
    
    
    def get_value(self,eval_args = None):
        """
        If no argument is passed in, return entire state.
    
        If an argument is passed in, this version of get_value assumes the argument is an index tuple
        and returns that element of state.
    
        This function may be overridden for particular parameters.
        """
        if eval_args is None:
                return self.value
        else:
                return self.value[eval_args]
    
    def get_state_counter(self):
        """
        This will be called by self's children when deciding whether they need to
        recompute their priors.
        """
        return self.state_counter
    
    
    def set_value(self,value):
    
        """
        The last value is being recorded so that DeterministicFunctions can cache their last
        evaluation and avoid recomputing if possible.
    
        I think we'll ultimately want to cache the last two values.
        """
        self.last_value = self.value
    
        # Increment the timestamp
        self.state_counter += 1
    
        # Set my value to the value that was passed in.
        if shape(value):
                 self.value = array(value)
        else:
                self.value = value
    
    

class Parameter(Node):
    
    def __init__(self, name, init_val, sampler=None, dist='normal', scale=None, observed = False, random=False, plot=True):
    
        """
        Initialize Node, etc.
        """
    
        # Extended children are nearest non-deterministic descendants.
        # See find_extended_children() method for more detail
        self.extended_children = None
    
        #Set current value of prior
        self.prior = None
    
        # Which sampler am I a member of?
        self._sampler = sampler
        if self._sampler is not None:
                self._sampler.add_parameter(self)
    
        """
        Initialize current value, record dimension, prepare proposal
        distribution, prepare _asf and _hyp, etc.
        """
    
    def compute_prior(self):
        """
        This method actually evaluates the prior distribution. It's called by
        self.get_prior as few times as possible, because it's likely to be
        one of the the slowest parts if not the slowest part of most MCMCs with
        moderately complicated Parameters.
    
        Must be implemented in each Parameter class, see Normal below.
        """
        pass
    
    def get_prior(self):
    
        """
        The goal here is to recompute the prior as few times as possible.
        The state counter variable of each parameter is incremented each time that
        parameter's value changes.
    
        In this method, self checks self.state_counter AND [parent.state_counter
        for parent in parents]. If all of these state counters match up with what
        was recorded last time self.compute_prior was called, there's no need to
        recompute the prior and this method returns the stored value.
    
        If something has changed since the last time self.compute_prior was called,
        self.compute_prior is called again.
    
        For maximum efficiency (measured in number of prior computations skipped), we'd
        really want to cache _2_ such sets of state counters and prior evaluations. The
        reason is that a Metropolis step can either be accepted or rejected.
        Think of the situation where a parameter C depends on parameters A and B, and
        B will make a Metropolis step, followed by A. B will call C.get_prior
        from a current value _and_ a proposed value. If C remembers its prior for _both_
        those values of B, then when A at its current value asks for C's prior C is
        guaranteed to not have to recompute.
    
        I haven't even tried to implement depth-2 caching, even with depth-1 caching I
        had to use exceptions because I got bogged down in all the conditionals.
    
        To lighten up the implementation, we should make this work if some elements of parents
        are simple numpy arrays rather than Parameters or Nodes. That removes the need for a
        special Constant parameter and reduces the number of Parameter declarations
        necessary.
        """
    
        try:
    
                # Has my state changed?
                if not self.state_counter_of_prior == self.state_counter:
                        raise NeedToRecompute
    
                for key in self.parents.keys():
    
                        """
                        Is the parent defined? This is an important check because, for example,
                        the Normal parameter can have a parent keyed by one of: 'tau' OR 'sigma' OR 'V',
                        and you want to check the slot that's actually in use.
                        """
                        if not self.parents[key] is None:
                                # If the parent is defined, is my snapshot of the parent's state up to date?
                                if not self.parents[key].get_state_counter() == self.parent_state_counters_of_prior[key]:
                                        raise NeedToRecompute
    
        """
        If necessary, recompute prior and increment state counters
        """
        except NeedToRecompute:
    
                # Call self.compute_prior if necessary
                self.current_prior = self.compute_prior()
    
                # Record self.state_counter and [parent.state_counter for parent in parents]
                self.state_counter_of_prior = self.state_counter
    
                for key in self.parents.keys():
                        if self.parents[key] is not None:
                                self.parent_state_counters_of_prior[key] = self.parents[key].get_state_counter()
    
    
        return self.current_prior
    
    
    
    def get_likelihood(self):
        """
        Return sum of log priors of self.extended_children, conditional on state of self
    
        Self's extended children is the set of parameters that depend on self either
        directly or via a sequence of DeterministicFunctions. See
        Parameter.find_extended_children for details.
        """
    
        # If my extended children haven't been found yet, go find them.
        if self.extended_children is None:
                self.find_extended_children ()
    
        """
        If self has no extended children, return 0. Otherwise return the sum of
        self's extended children's priors.
        """
        if self.extended_children == set([]):
                return 0
        else:
                return sum([child.get_prior() for child in self.extended_children])
    
    
    
    def find_extended_children(self):
    
        """
        Self may have children that are instances of DeterministicFunction.
        If that's the case, self.get_likelihood doesn't want to ask those children
        for their priors.
    
        What self.get_likelihood wants to do is query all the non-deterministic
        parameters that either depend on self directly or depend on a
        DeterministicFunction (or possibly a sequence of DeterministicFunctions)
        that takes self as an argument.
    
        This method finds those non-deterministic parameters, which I've called the
        extended children of self (there's got to be a standard word for them out there
        somewhere, but I haven't been able to find it).
        """
    
        # Start with my immediate children, deterministic or no.
        if self.extended_children is None:
                self.extended_children = self.children
    
        # If none of my extended children is a DeterministicFunction, stop.
        if not any([isinstance(child,DeterministicFunction) for child in self.extended_children]):
                return
    
    
        else:
                # Go through each of my extended children
                for child in self.extended_children:
    
                        # If child is not a DeterministicFunction, move on to the next child.
                        if isinstance(child,DeterministicFunction):
    
                                #If child is a DeterministicFunction, replace child with child.children
                                self.extended_children.update(child.children)
                                self.extended_children.discard(child)
    
                """
                Recur. The reason is some children that are DeterministicFunctions may themselves
                have children that are DeterministicFunctions, so the nearest non-deterministic
                descendants of self may be more than one 'generation' removed.
                """
                self.find_extended_children()
    
        return
    
    
    
    def metropolis_step(self, debug=False):
    
        """
        When self.metropolis_step is called, self makes a jump according to the
        Metropolis algorithm.
        """
    
        """
        The flag 'observed' tells whether self holds data. If it does,
        then self.metropolis_step has been called carelessly and nothig should be
        done.
        """
        if self.observed:
                return
    
    
        """
        These calls to self.get_prior and self.extended_children.get_prior via
        self.get_likelihood won't actually result in recomputation unless
        it's necessary.
        """
        try:
                old_prior = self.get_prior()
                old_like = self.get_likelihood()
        except ParametrizationError, msg:
                print msg
                sys.exit(1)
    
        """
        Propose new values using a random walk algorithm, according to
        the proposal distribution specified:
    
        x(t+1) = x(t) + e(t)
    
        where e ~ proposal(hyperparameters)
        """
        self.sample_candidate()
    
        """
        The following calls will nearly always result in computation, but there's
        no way around that.
        """
        try:
                new_prior = self.get_prior()
                new_like = self.get_likelihood()
        except ParametrizationError, msg:
                print msg
                sys.exit(1)
    
        """Accept or reject proposed parameter values"""
    
        # Reject bogus results
        if str(new_like) == 'nan' or new_like == -inf or str(new_prior) == 'nan' or new_prior == -inf:
                self.revert()
                self._rejected += 1
                return
    
        # Compute log of probability ratio.
        logp_difference = new_like - old_like + new_prior - old_prior
    
        # Test
        try:
    
                """
                If the jump is accepted, do nothing, because self.value has been set
                to the proposed value by self.sample_candidate.
                """
                if log(random_number()) <= logp_difference:
                        pass
    
                """
                If the jump is rejected, call the method self.revert instead of just doing
                self.set_value(self.last_value) or something like that. The reason is given below.
                """
                else:
                        self.revert()
                        self._rejected += 1
    
        except ParameterError:
                print self.name , ': ', msg
                sys.exit(1)
    
    def revert(self):
    
        """
        Revert sets self.value to self.last_value AND decrements self.state_counter,
        so that self's extended children know that they've seen this situation before.
        """
    
        if self.last_value is not None:
                self.value = self.last_value
                self.last_value = None
                self.state_counter -= 1
        else:
                raise ParameterError, self.name + ": can't revert, last_value not defined"
    
    
    
    
    class DeterministicFunction(Node):
    """
    DeterministicFunction is parameter whose value is a function of its parents'
    values, but not of any random variables. Pass this function in as eval_fun.
    Pass the argument keys as parent_keys.
    
    For example, to implement parameter z in z = x^2 + y^2,
    
    z = DeterministicFunction("z",  lambda x,y : x ** 2 + y ** 2,  ("x","y") )
    z.add_parent(x,"x")
    z.add_parent(y,"y")
    
    and you're ready to go... this would be another fine place to work on
    'lightening up' the implementation!
    
    
    DeterministicFunction tries to avoid recomputing its value in the same way as
    Parameter tries to avoid recomputing its prior, using state counters. Seems like this
    could become important if, say, someone writes a nonlinear ODE or somesuch as a
    DeterministicFunction.
    """
    
    def __init__(self, name, eval_fun, parent_keys, sampler=None, observed=False, random=False, plot=False, output_shape = ()):
    
        Node.__init__(self, name, sampler, None, observed, output_shape, plot)
    
        self.eval_fun = vectorize(eval_fun)
        self.parent_keys = parent_keys
    
        self.args = None
        self.value = None
    
        for key in parent_keys:
                self.parent_state_counters_of_prior[key] = None
    
    
    def get_value(self,eval_args = None):
    
        # Make sure every key in self.parent_keys has an actual parent associated.
        if any([not self.parents.has_key(key) for key in self.parent_keys]):
                raise ParametrizationError
    
        """
        Here comes the timestamp checking. As with Parameter, the state counters should be
        cached 2 deep instead of 1 deep, as I have done.
        """
        try:
                # If my value has never been computed, recompute.
                if self.value is None: raise NeedToRecompute
    
                # If any of my parents have changed value since the last time my value was computed, recompute.
                for key in self.parent_keys:
                        if not self.parents[key].get_state_counter() == self.parent_state_counters_of_prior[key]: raise NeedToRecompute
    
    
        except NeedToRecompute:
                # If necessary, recompute value
                self.set_value(self.compute_value())
    
                # Record my state counter, for my children's sake (see Parameter.get_prior).
                self.state_counter_of_prior = self.state_counter
    
                # Record my parents' state counters
                for key in self.parents.keys():
                        self.parent_state_counters_of_prior[key] = self.parents[key].get_state_counter()
    
        """
        If no argument is passed in, return entire value.
    
        If an argument is passed in, this version of get_value assumes the argument is an index tuple
        and returns that element of self.value.
    
        We should probably figure out some way for the user to override this little bit of code without
        having to copy and paste all the timestamp stuff.
        """
        if eval_args is None:
                return self.value
        else:
                return self.value[eval_args]
    
    
    
    def get_state_counter(self):
    
        """
        OK. This method is being called by self's extended children, because they're trying to
        figure out whether they need to recompute their prior.
    
        That means self needs to check whether self's parents have changed state since the last time
        self.s compute_value was called, since self's value should always reflect self's parents'
        values.
    
        This method calls self.get_value(), which will check self's parents' state counters and,
        if necessary, recompute self's own value and increment self's state counter.
        """
        self.get_value()
    
        # Return current state counter
        return self.state_counter
    
    
    
    def compute_value(self):
        """
        Evaluates the function that was passed to my constructor based on self's parents' values.
    
        This method is called by self.get_value only when necessary.
    
        It seems like this implementation runs the risk of getting arguments out of order...
        """
    
        self.args = ([self.parents[key].get_value(self.parent_eval_args [key]) for key in self.parent_keys])
        return self.eval_fun(*self.args)
    
    
    
    def set_value(self):
        # My value is directly tied to the value of my parents, so the user
        # should never try to set my value.
        print "Warning: DeterministicFunction ",name,"'s value set by the user"
    

class SubSampler:
    
    def __init__(self, sampler = None, parameters = None, debug = False):
    
        """
        SubSamplers keep track of their children and extended children just like
        Parameters do. That's because they're usually embedded in larger models,
        so they need to keep track of the rest of the model when evaluating jumps.
        """
        self.children = set()
    
    
    
        # Add all the parameters that were passed into the constructor to myself.
        for parameter in parameters:
                self.add_parameter(parameter)
    
        """
        This doesn't feel like a good long-term solution to the problem of
        DeterministicFunctions, maybe SubSampler should maintain a special set containing
        the parameters that can be updated?
    
        What I have in mind here is as follows: The Joint SubSampler that we're thinking
        about shouldn't try to propose new values for DeterministicFunctions, it should
        propose new values for actual unknown parameters. The DeterministicFunctions
        will be updated automatically when the parameters start trying to compute their
        priors.
    
        However, if we use SubSamplers to package parts of models that get handled by
        different processes, it will often make sense to include DeterministicFunctions.
        """
        if any(self.parameters) is DeterministicFunction:
                print "Error, deterministic parameters cannot be added to a sampler"
                raise ParametrizationError
    
    
    
    def add_parameter(self,new_parameter):
    
        """
        This deserves more thought, see the last paragraph in __init__.
        """
        if new_parameter is DeterministicFunction:
                print "Error, deterministic function " + new_parameter.name + " cannot be added to a subsampler."
                raise ParametrizationError
        else:
                """
                Add the new parameter to self.parameters, and also add its children to self.children,
                provided its children aren't elements of self.parameters.
    
                If the new parameter happens to be in self.children already, remove it.
                """
                self.parameters.update([new_parameter])
                self.children.update(new_parameter.children).difference(parameters)
                self.children -= set(new_parameter)
    
    
    
    def get_prior(self):
        """
        'Prior' is a funny word for this, really, it's just named to be consistent with the
        analogous method in Parameter. This method returns the log joint probability of all of
        self's member Parameters, conditional on their parents.
    
        No complicated monkeyshines with state counters are necessary here, each parameter will make
        its own decision about whether to recompute its prior.
        """
        return sum([parameter.get_prior() for parameter in self.parameters])
    
    def find_extended_children(self):
        """
        This method is exactly like the analogous method in Parameter, except the last bit:
        """
        if self.extended_children is None:
                # Start with my children
                self.extended_children = self.children
    
        # If none of my children is a DeterministicFunction, stop
        if not any([isinstance(child,DeterministicFunction) for child in self.extended_children]):
                return
    
        else:
                # Go through my extended children
                for child in self.extended_children:
    
                        # If the current child is a DeterministicFunction, replace it with its children.
                        if isinstance(child,DeterministicFunction):
                                self.extended_children.update(child.children)
                                self.extended_children.discard(child)
    
                # If any of my extended children are member parameters of me, remove them.
                self.extended_chilren -= self.parameters
    
                #Recur
                self.find_extended_children
    
    
    
    def get_likelihood(self):
        """
        Return sum oflog priors of self's extended children, conditional on state of self.
        This is just like Parameter.get_likelihood.
        """
    
        # If my extended children haven't been found, go find them.
        if self.extended_children is None:
                self.find_extended_children()
    
        # If I have no extended children, return 0.
        if self.extended_children == set([]):
                return 0
    
        # Otherwise return the sum of my extended children's priors.
        else:
                return sum([child.get_prior() for child in self.extended_children])
    
    
    
    def step(self):
        """
        Overrideable for individual SubSamplers.
        The default behavior is one-at-a-time Metropolis sampling.
        """
        for parameter in self.parameters:
                parameter.metropolis_step()
    
    # And so on.
    def tally(self):
        for parameter in self.parameters:
                parameter.tally()
    
    def tune(self):
        for parameter in self.parameters:
                parameter.tune()
    

class Sampler:
    """
    We've got decisions to make here. It should be possible for Sampler.parameter (in PyMC currently),
    Sampler.add_parameter (new here), and Parameter to work in harmony. We just have to figure out how
    to get all of them to put a reference to the parameter on the base namespace and in Sampler's
    parameters dictionary as in PyMC currently. (We also have to resolve the naming conflict between that
    dictionary and the set self.parameters below).
    
    However, I can't see any way Sampler.sample could be able to provide PyMC's current
    implementation _and_ what I'm asking it to do below. That means we would have to either
    add a method to Sampler for sampling in the new mode or write a new class (maybe Model) that
    samples in the new mode.
    
    If we do write a new class, we should of course keep all the nice support that Sampler currently
    provides.
    """
    
    def __init__(self, plot_format='png', plot_backend='TkAgg'):
    
        self.nodes = {}
        self.parameters = set()
         self.subsamplers = set()
    
        # Deviance node, goodness-of-fit flag, plotter, etc.
    
    def add_parameter(self,new_parameter):
        # Needs to play well with Sampler.parameter(), too
        self.parameters.add(new_parameter)
    
    # Same thing for nodes.
    
    def add_subsampler(self,new_subsampler):
        self.subsamplers.add(new_subsampler)
    
    def sample(self, iterations, burn=0, thin=1, tune=True, tune_interval=100, divergence_threshold=1e10, verbose=True, plot=True, debug=False):
        """
        I haven't thought this through very hard, but the main idea is that Sampler begins by
        scooping up all the parameters that aren't members of a SubSampler already and adding
        them to a new, anonymous SubSampler that does one-at-a-time Metropolis sampling.
    
        Then, Sampler tells each of its member SubSamplers to step, tune, etc. sequentially.
    
        Sampler is responsible for managing the loop, plotting, and compiling model-level
        goodness-of-fit statistics as it currently does in PyMC.
        """
    
        #Find all parameters that aren't members of a subsampler
        self.lone_parameters = self.parameters
        for subsampler in self.subsamplers:
                self.lone_parameters.difference(subsampler.parameters)
    
        #Gather the lone parameters into a one-at-a-time subsampler
        if len(self.lone_parameters) > 0:
                self.add_subsampler(SubSampler(self.lone_parameters))
    
        #Tell all the subsamplers to step, tune, etc.
        for iteration in range(iterations):
                for subsampler in subsamplers:
                        subsampler.sample()
                        subsampler.tune()
    
    

class Normal(Parameter):
    """
    As David said, this is a really heavy implementation. I was trying to provide support for
    all three standard parametrizations of the normal distribution without forcing the
    user to use DeterministicFunctions, but there must be a nicer way.
    
    A good implementational model for this class would probably generalize pretty readily to
    other Parameter subclasses with multiple standard parametrizations.
    """
    
    def __init__(self, name, init_val=None, model=None, dist='normal', scale=None, observed = False, random=False, plot=True):
    
        Parameter.__init__(self, name, init_val, model, dist, scale, observed, random, plot)
    
        self.parents["mu"] = None
        self.parents["V"] = None
        self.parents["sigma"] = None
        self.parents["tau"] = None
        self.mu = None
        self.tau = None
        self.x = None
    
        # vectorize fnormal
        self.vfnormal = vectorize(fnormal)
    
    
    
    def compute_prior(self):
    
    
        # Make sure mu is defined
        if self.parents["mu"] is None:
                raise ParametrizationError, self.name + ': my mu parameter is missing.'
    
        # Make sure V, tau, or sigma is defined.
        elif ( self.parents["V"] is None and self.parents["sigma"] is None and self.parents["tau"] is None ):
                raise ParametrizationError, self.name + ": I don't have a V, sigma, or tau parameter."
    
        # Make sure only one of the variance parameters is defined.
        elif sum([ self.parents["V"] is None, self.parents["sigma"] is None, self.parents["tau"] is None ]) < 2:
                raise ParametrizationError, self.name + ": I only want a V, sigma, OR tau parameter, but have more than one of these."
    
        """
        Most of the extra pork comes from checking that the parents are all either
        scalars or of the same shape as self. Come to think of it, we should probably
        just leave that to Chris's normal_like, etc. methods somehow.
        """
    
        # Retrieve mean
        shape_now = shape(self.parents["mu"].get_value(self.parent_eval_args["mu"]))
        if shape_now == ():
                 self.mu = self.parents["mu"].get_value(self.parent_eval_args["mu"])
        elif not shape_now == shape(self.current_value):
                raise ParametrizationError, self.name + ": my mu parameter " + self.parents["mu"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
        else:
                 self.mu = reshape(self.parents["mu"].get_value(self.parent_eval_args["mu"]),-1)
    
        # Retrieve sigma, tau, or V, depending on which parent is defined
        if self.parents["V"]:
                try:
                        self.parents["V"].constrain(lower=0)
                except ParameterError:
                        return -inf
                shape_now = shape(self.parents["V"].get_value(self.parent_eval_args["V"]))
                if shape_now == ():
                        self.tau = 1. /self.parents["V"].get_value( self.parent_eval_args["V"])
                elif not shape_now == shape(self.current_value):
                        raise ParametrizationError, self.name + ": my V parameter " + self.parents["V"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
                else:
                        self.tau = 1. / reshape( self.parents["V"].get_value(self.parent_eval_args["V"]),-1)
    
        if self.parents["sigma"]:
                shape_now = shape(self.parents["sigma"].get_value( self.parent_eval_args["sigma"]))
                if shape_now == ():
                        self.tau = 1. / self.parents["sigma"].get_value(self.parent_eval_args["sigma"]) ** 2.
                elif not shape_now == shape(self.current_value):
                        raise ParametrizationError, self.name + ": my sigma parameter " + self.parents ["sigma"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
                else:
                        self.tau = 1. / reshape( self.parents["sigma"].get_value(self.parent_eval_args["sigma"]),-1)
                        self.tau *= self.tau
    
    
        if self.parents["tau"]:
                try:
                        self.parents["tau"].constrain(lower=0)
                except ParameterError:
                        return -inf
                shape_now = shape( self.parents["tau"].get_value(self.parent_eval_args["tau"]))
                if shape_now == ():
                        self.tau = self.parents["tau"].get_value(self.parent_eval_args ["tau"])
                elif not shape_now == shape(self.current_value):
                        raise ParametrizationError, self.name + ": my tau parameter " + self.parents["tau"].name + "'s shape is " + str(shape_now) + " but mine is " + str(shape(self.current_value))
                else:
                        self.tau = reshape( self.parents["tau"].get_value(self.parent_eval_args["tau"]),-1)
    
    
        """
        It's important to use get_value() here rather than accessing self.value directly,
        because get_value() checks timestamps.
        """
        if self.dim:
                self.x = ravel(self.get_value())
        else:
                 self.x = self.get_value()
    
        self.x -= self.mu
        self.x *= self.x
        self.x *= self.tau
        logp = .5 * (sum(log(self.tau)) - sum( self.x))
        return logp



