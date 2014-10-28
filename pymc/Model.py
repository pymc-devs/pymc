"""
Base classes Model and Sampler are defined here.
"""

# Changeset history
# 22/03/2007 -DH- Added methods to query the StepMethod's state and pass it to database.
# 20/03/2007 -DH- Separated Model from Sampler. Removed _prepare().
# Commented __setattr__ because it breaks properties.

__docformat__ = 'reStructuredText'
__all__ = ['Model', 'Sampler']

""" Summary"""

from numpy import zeros, floor
from numpy.random import randint
from . import database
from .PyMCObjects import Stochastic, Deterministic, Node, Variable, Potential
from .Container import Container, ObjectContainer
import sys
import os
from copy import copy
from threading import Thread
from .Node import ContainerBase
from time import sleep
import pdb
from . import utils
import warnings
import traceback
import itertools

from .six import print_, reraise

GuiInterrupt = 'Computation halt'
Paused = 'Computation paused'


class EndofSampling(Exception):
    pass


class Model(ObjectContainer):

    """
    The base class for all objects that fit probability models. Model is initialized with:

      >>> A = Model(input, verbose=0)

      :Parameters:
        - input : module, list, tuple, dictionary, set, object or nothing.
            Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
            If nothing, all nodes are collected from the base namespace.

    Attributes:
      - deterministics
      - stochastics (with observed=False)
      - data (stochastic variables with observed=True)
      - variables
      - potentials
      - containers
      - nodes
      - all_objects
      - status: Not useful for the Model base class, but may be used by subclasses.

    The following attributes only exist after the appropriate method is called:
      - moral_neighbors: The edges of the moralized graph. A dictionary, keyed by stochastic variable,
        whose values are sets of stochastic variables. Edges exist between the key variable and all variables
        in the value. Created by method _moralize.
      - extended_children: The extended children of self's stochastic variables. See the docstring of
        extend_children. This is a dictionary keyed by stochastic variable.
      - generations: A list of sets of stochastic variables. The members of each element only have parents in
        previous elements. Created by method find_generations.

    Methods:
       - sample_model_likelihood(iter): Generate and return iter samples of p(data and potentials|model).
         Can be used to generate Bayes' factors.

    :SeeAlso: Sampler, MAP, NormalApproximation, weight, Container, graph.
    """

    def __init__(self, input=None, name=None, verbose=-1):
        """Initialize a Model instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
        """

        # Get stochastics, deterministics, etc.
        if input is None:
            import warnings
            warnings.warn(
                'The MCMC() syntax is deprecated. Please pass in nodes explicitly via M = MCMC(input).')
            import __main__
            __main__.__dict__.update(self.__class__.__dict__)
            input = __main__

        ObjectContainer.__init__(self, input)

        if name is not None:
            self.__name__ = name
        self.verbose = verbose

    def _get_generations(self):
        if not hasattr(self, '_generations'):
            self._generations = utils.find_generations(self)
        return self._generations
    generations = property(_get_generations)

    def draw_from_prior(self):
        """
        Sets all variables to random values drawn from joint 'prior', meaning contributions
        of data and potentials to the joint distribution are not considered.
        """

        for generation in self.generations:
            for s in generation:
                s.random()

    def seed(self):
        """
        Seed new initial values for the stochastics.
        """

        for generation in self.generations:
            for s in generation:
                try:
                    if s.rseed is not None:
                        value = s.random(**s.parents.value)
                except:
                    pass

    def get_node(self, node_name):
        """Retrieve node with passed name"""
        for node in self.nodes:
            if node.__name__ == node_name:
                return node


class Sampler(Model):

    """
    The base class for all objects that fit probability models using Monte Carlo methods.
    Sampler is initialized with:

      >>> A = Sampler(input, db, output_path=None, verbose=0)

      :Parameters:
        - input : module, list, tuple, dictionary, set, object or nothing.
            Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
            If nothing, all nodes are collected from the base namespace.
        - db : string
            The name of the database backend that will store the values
            of the stochastics and deterministics sampled during the MCMC loop.

    Inherits all methods and attributes from Model. Subclasses must either define the _loop method:

        - _loop(self, *args, **kwargs): Can be called after a sampling run is interrupted
            (by pausing, halting or a KeyboardInterrupt) to continue the sampling run.
            _loop must be able to handle KeyboardInterrupts gracefully, and should monitor
            the sampler's status periodically. Available status values are:
            - 'ready': Ready to sample.
            - 'paused': A pause has been requested, or the sampler is paused. _loop should return control
                as soon as it is safe to do so.
            - 'halt': A halt has been requested, or the sampler is stopped. _loop should call halt as soon
                as it is safe to do so.
            - 'running': Sampling is in progress.

    Or define a draw() method, which draws a single sample from the posterior. Subclasses may also want
    to override the default sample() method.

    :SeeAlso: Model, MCMC.
    """

    def __init__(self, input=None, db='ram', name='Sampler',
                 reinit_model=True, calc_deviance=False, verbose=0, **kwds):
        """Initialize a Sampler instance.

        :Parameters:
          - input : module, list, tuple, dictionary, set, object or nothing.
              Model definition, in terms of Stochastics, Deterministics, Potentials and Containers.
              If nothing, all nodes are collected from the base namespace.
          - db : string
              The name of the database backend that will store the values
              of the stochastics and deterministics sampled during the MCMC loop.
          - reinit_model : bool
              Flag for reinitialization of Model superclass.
          - calc_deviance : bool
              Flag for calculating model deviance.
          - **kwds :
              Keywords arguments to be passed to the database instantiation method.
        """

        # Initialize superclass
        if reinit_model:
            Model.__init__(self, input, name, verbose)

        # Initialize deviance, if asked
        if calc_deviance:
            self._funs_to_tally = {'deviance': self._sum_deviance}
        else:
            self._funs_to_tally = {}

        # Specify database backend and save its keywords
        self._db_args = kwds
        self._assign_database_backend(db)

        # Flag for model state
        self.status = 'ready'

        self._current_iter = None
        self._iter = None

        self._state = ['status', '_current_iter', '_iter']

        if hasattr(db, '_traces'):
            # Put traces on objects
            for v in self._variables_to_tally:
                v.trace = self.db._traces[v.__name__]

    def _sum_deviance(self):
        # Sum deviance from all stochastics

        return -2 * sum([v.get_logp() for v in self.observed_stochastics])

    def sample(self, iter, length=None, verbose=0):
        """
        Draws iter samples from the posterior.
        """
        self._cur_trace_index = 0
        self.max_trace_length = iter
        self._iter = iter
        self.verbose = verbose or 0
        self.seed()

        # Assign Trace instances to tallyable objects.
        self.db.connect_model(self)

        # Initialize database -> initialize traces.
        if length is None:
            length = iter
        self.db._initialize(self._funs_to_tally, length)

        # Put traces on objects
        for v in self._variables_to_tally:
            v.trace = self.db._traces[v.__name__]

        # Loop
        self._current_iter = 0
        self._loop()
        self._finalize()

    def _finalize(self):
        """Reset the status and tell the database to finalize the traces."""
        if self.status in ['running', 'halt']:
            if self.verbose > 0:
                print_('\nSampling finished normally.')
            self.status = 'ready'

        self.save_state()
        self.db._finalize()

    def _loop(self):
        """
        _loop(self, *args, **kwargs)

        Can be called after a sampling run is interrupted (by pausing, halting or a
        KeyboardInterrupt) to continue the sampling run.

        _loop must be able to handle KeyboardInterrupts gracefully, and should monitor
        the sampler's status periodically. Available status values are:
        - 'ready': Ready to sample.
        - 'paused': A pause has been requested, or the sampler is paused. _loop should return control
            as soon as it is safe to do so.
        - 'halt': A halt has been requested, or the sampler is stopped. _loop should call halt as soon
            as it is safe to do so.
        - 'running': Sampling is in progress.
        """
        self.status = 'running'

        try:
            while self._current_iter < self._iter and not self.status == 'halt':
                if self.status == 'paused':
                    break

                i = self._current_iter

                self.draw()
                self.tally()

                if not i % 10000 and self.verbose > 0:
                    print_('Iteration ', i, ' of ', self._iter)
                    sys.stdout.flush()

                self._current_iter += 1

        except KeyboardInterrupt:
            self.status = 'halt'

        if self.status == 'halt':
            self._halt()

    def draw(self):
        """
        Either draw() or _loop() must be overridden in subclasses of Sampler.
        """
        pass

    def stats(self, variables=None, alpha=0.05, start=0,
              batches=100, chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
        """
        Statistical output for variables.

        :Parameters:
        variables : iterable
          List or array of variables for which statistics are to be
          generated. If it is not specified, all the tallied variables
          are summarized.

        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.

        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.

        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        """

        # If no names provided, run them all
        if variables is None:
            variables = self._variables_to_tally
        else:
            variables = [v for v in self.variables if v.__name__ in variables]

        stat_dict = {}

        # Loop over nodes
        for variable in variables:
            # Plot object
            stat_dict[variable.__name__] = self.trace(
                variable.__name__).stats(alpha=alpha, start=start,
                                         batches=batches, chain=chain, quantiles=quantiles)

        return stat_dict

    def write_csv(
        self, filename, variables=None, alpha=0.05, start=0, batches=100,
            chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
        """
        Save summary statistics to a csv table.

        :Parameters:

        filename : string
          Filename to save output.

        variables : iterable
          List or array of variables for which statistics are to be
          generated. If it is not specified, all the tallied variables
          are summarized.

        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.

        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.

        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        """

        # Append 'csv' suffix if there is no suffix on the filename
        if filename.find('.') == -1:
            filename += '.csv'

        outfile = open(filename, 'w')

        # Write header to file
        header = 'Parameter, Mean, SD, MC Error, Lower 95% HPD, Upper 95% HPD, '
        header += ', '.join(['q%s' % i for i in quantiles])
        outfile.write(header + '\n')

        stats = self.stats(
            variables=variables,
            alpha=alpha,
            start=start,
            batches=batches,
            chain=chain,
            quantiles=quantiles)

        if variables is None:
            variables = sorted(stats.keys())

        buffer = str()
        for param in variables:

            values = stats[param]

            try:
                # Multivariate node
                shape = values['mean'].shape
                indices = list(itertools.product(*[range(i) for i in shape]))

                for i in indices:
                    buffer += self._csv_str(param, values, quantiles, i)

            except AttributeError:
                # Scalar node
                buffer += self._csv_str(param, values, quantiles)

        outfile.write(buffer)

        outfile.close()

    def _csv_str(self, param, stats, quantiles, index=None):
        """Support function for write_csv"""

        buffer = param
        if not index:
            buffer += ', '
        else:
            buffer += '_' + '_'.join([str(i) for i in index]) + ', '

        for stat in ('mean', 'standard deviation', 'mc error'):
            buffer += str(stats[stat][index]) + ', '

        # Index to interval label
        iindex = [key.split()[-1] for key in stats.keys()].index('interval')
        interval = list(stats.keys())[iindex]
        buffer += ', '.join(stats[interval].T[index].astype(str))

        # Process quantiles
        qvalues = stats['quantiles']
        for q in quantiles:
            buffer += ', ' + str(qvalues[q][index])

        return buffer + '\n'

    def summary(self, variables=None, alpha=0.05, start=0, batches=100,
                chain=None, roundto=3):
        """
        Generate a pretty-printed summary of the model's variables.

        :Parameters:
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.

        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.

        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.

        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).

        roundto : int
          The number of digits to round posterior statistics.

        quantiles : tuple or list
          The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).
        """

        # If no names provided, run them all
        if variables is None:
            variables = self._variables_to_tally
        else:
            variables = [
                self.__dict__[
                    i] for i in variables if self.__dict__[
                        i] in self._variables_to_tally]

        # Loop over nodes
        for variable in variables:
            variable.summary(
                alpha=alpha, start=start, batches=batches, chain=chain,
                roundto=roundto)

    # Property --- status : the sampler state.
    def status():
        doc = \
            """Status of sampler. May be one of running, paused, halt or ready.
          - `running` : The model is currently sampling.
          - `paused` : The model has been interrupted during sampling. It is
            ready to be restarted by `continuesample`.
          - `halt` : The model has been interrupted. It cannot be restarted.
            If sample is called again, a new chain will be initiated.
          - `ready` : The model is ready to sample.
        """

        def fget(self):
            return self.__status

        def fset(self, value):
            if value in ['running', 'paused', 'halt', 'ready']:
                self.__status = value
            else:
                raise AttributeError(value)
        return locals()
    status = property(**status())

    def _assign_database_backend(self, db):
        """Assign Trace instance to stochastics and deterministics and Database instance
        to self.

        :Parameters:
          - `db` : string, Database instance
            The name of the database module (see below), or a Database instance.

        Available databases:
          - `no_trace` : Traces are not stored at all.
          - `ram` : Traces stored in memory.
          - `txt` : Traces stored in memory and saved in txt files at end of
                sampling.
          - `sqlite` : Traces stored in sqlite database.
          - `hdf5` : Traces stored in an HDF5 file.
        """
        # Objects that are not to be tallied are assigned a no_trace.Trace
        # Tallyable objects are listed in the _nodes_to_tally set.

        no_trace = getattr(database, 'no_trace')
        self._variables_to_tally = set()
        for object in self.stochastics | self.deterministics:

            if object.keep_trace:
                self._variables_to_tally.add(object)
                try:
                    if object.mask is None:
                        # Standard stochastic
                        self._funs_to_tally[object.__name__] = object.get_value
                    else:
                        # Has missing values, so only fetch stochastic elements
                        # using mask
                        self._funs_to_tally[
                            object.__name__] = object.get_stoch_value
                except AttributeError:
                    # Not a stochastic object, so no mask
                    self._funs_to_tally[object.__name__] = object.get_value
            else:
                object.trace = no_trace.Trace(object.__name__)

        check_valid_object_name(self._variables_to_tally)

        # If not already done, load the trace backend from the database
        # module, and assign a database instance to Model.
        if isinstance(db, str):
            if db in dir(database):
                module = getattr(database, db)

                # Assign a default name for the database output file.
                if self._db_args.get('dbname') is None:
                    self._db_args['dbname'] = self.__name__ + '.' + db

                self.db = module.Database(**self._db_args)
            elif db in database.__modules__:
                raise ImportError(
                    'Database backend `%s` is not properly installed. Please see the documentation for instructions.' % db)
            else:
                raise AttributeError(
                    'Database backend `%s` is not defined in pymc.database.' % db)
        elif isinstance(db, database.base.Database):
            self.db = db
            self.restore_sampler_state()
        else:   # What is this for? DH.
            self.db = db.Database(**self._db_args)

    def pause(self):
        """Pause the sampler. Sampling can be resumed by calling `icontinue`.
        """
        self.status = 'paused'
        # The _loop method will react to 'paused' status and stop looping.
        if hasattr(
                self, '_sampling_thread') and self._sampling_thread.isAlive():
            print_('Waiting for current iteration to finish...')
            while self._sampling_thread.isAlive():
                sleep(.1)

    def halt(self):
        """Halt a sampling running in another thread."""
        self.status = 'halt'
        # The _halt method is called by _loop.
        if hasattr(
                self, '_sampling_thread') and self._sampling_thread.isAlive():
            print_('Waiting for current iteration to finish...')
            while self._sampling_thread.isAlive():
                sleep(.1)

    def _halt(self):
        print_('Halting at iteration ', self._current_iter, ' of ', self._iter)
        self.db.truncate(self._cur_trace_index)
        self._finalize()

    #
    # Tally
    #
    def tally(self):
        """
        tally()

        Records the value of all tracing variables.
        """
        if self.verbose > 2:
            print_(self.__name__ + ' tallying.')
        if self._cur_trace_index < self.max_trace_length:
            self.db.tally()

        self._cur_trace_index += 1
        if self.verbose > 2:
            print_(self.__name__ + ' done tallying.')

    def commit(self):
        """
        Tell backend database to commit.
        """

        self.db.commit()

    def isample(self, *args, **kwds):
        """
        Samples in interactive mode. Main thread of control stays in this function.
        """
        self._exc_info = None
        out = kwds.pop('out', sys.stdout)
        kwds['progress_bar'] = False

        def samp_targ(*args, **kwds):
            try:
                self.sample(*args, **kwds)
            except:
                self._exc_info = sys.exc_info()

        self._sampling_thread = Thread(
            target=samp_targ,
            args=args,
            kwargs=kwds)
        self.status = 'running'
        self._sampling_thread.start()
        self.iprompt(out=out)

    def icontinue(self):
        """
        Restarts thread in interactive mode
        """
        if self.status != 'paused':
            print_(
                "No sampling to continue. Please initiate sampling with isample.")
            return

        def sample_and_finalize():
            self._loop()
            self._finalize()

        self._sampling_thread = Thread(target=sample_and_finalize)
        self.status = 'running'
        self._sampling_thread.start()
        self.iprompt()

    def iprompt(self, out=sys.stdout):
        """Start a prompt listening to user input."""

        cmds = """
        Commands:
          i -- index: print current iteration index
          p -- pause: interrupt sampling and return to the main console.
                      Sampling can be resumed later with icontinue().
          h -- halt:  stop sampling and truncate trace. Sampling cannot be
                      resumed for this chain.
          b -- bg:    return to the main console. The sampling will still
                      run in a background thread. There is a possibility of
                      malfunction if you interfere with the Sampler's
                      state or the database during sampling. Use this at your
                      own risk.
        """

        print_("""==============
 PyMC console
==============

        PyMC is now sampling. Use the following commands to query or pause the sampler.
        """, file=out)
        print_(cmds, file=out)

        prompt = True
        try:
            while self.status in ['running', 'paused']:
                    # sys.stdout.write('pymc> ')
                if prompt:
                    out.write('pymc > ')
                    out.flush()

                if self._exc_info is not None:
                    a, b, c = self._exc_info
                    reraise(a, b, c)

                cmd = utils.getInput().strip()
                if cmd == 'i':
                    print_(
                        'Current iteration: %i of %i' %
                        (self._current_iter, self._iter), file=out)
                    prompt = True
                elif cmd == 'p':
                    self.status = 'paused'
                    break
                elif cmd == 'h':
                    self.status = 'halt'
                    self._halt()
                    break
                elif cmd == 'b':
                    return
                elif cmd == '\n':
                    prompt = True
                    pass
                elif cmd == '':
                    prompt = False
                else:
                    print_('Unknown command: ', cmd, file=out)
                    print_(cmds, file=out)
                    prompt = True

        except KeyboardInterrupt:
            if not self.status == 'ready':
                self.status = 'halt'
                self._halt()

        if self.status == 'ready':
            print_("Sampling terminated successfully.", file=out)
        else:
            print_('Waiting for current iteration to finish...', file=out)
            while self._sampling_thread.isAlive():
                sleep(.1)
            print_('Exiting interactive prompt...', file=out)
            if self.status == 'paused':
                print_(
                    'Call icontinue method to continue, or call halt method to truncate traces and stop.',
                    file=out)

    def get_state(self):
        """
        Return the sampler's current state in order to
        restart sampling at a later time.
        """
        state = dict(sampler={}, stochastics={})
        # The state of the sampler itself.
        for s in self._state:
            state['sampler'][s] = getattr(self, s)

        # The state of each stochastic parameter
        for s in self.stochastics:
            state['stochastics'][s.__name__] = s.value
        return state

    def save_state(self):
        """
        Tell the database to save the current state of the sampler.
        """
        try:
            self.db.savestate(self.get_state())
        except:
            print_('Warning, unable to save state.')
            print_('Error message:')
            traceback.print_exc()

    def restore_sampler_state(self):
        """
        Restore the state of the sampler and to
        the state stored in the database.
        """

        state = self.db.getstate() or {}

        # Restore sampler's state
        sampler_state = state.get('sampler', {})
        self.__dict__.update(sampler_state)

        # Restore stochastic parameters state
        stoch_state = state.get('stochastics', {})
        for sm in self.stochastics:
            try:
                sm.value = stoch_state[sm.__name__]
            except:
                warnings.warn(
                    'Failed to restore state of stochastic %s from %s backend' %
                    (sm.__name__, self.db.__name__))
                # print_('Error message:')
                # traceback.print_exc()

    def remember(self, chain=-1, trace_index=None):
        """
        remember(chain=-1, trace_index = randint(trace length to date))

        Sets the value of all tracing variables to a value recorded in
        their traces.
        """
        if trace_index is None:
            trace_index = randint(self._cur_trace_index)

        for variable in self._variables_to_tally:
            if isinstance(variable, Stochastic):
                try:
                    variable.value = self.trace(
                        variable.__name__,
                        chain=chain)[trace_index]
                except:
                    cls, inst, tb = sys.exc_info()
                    warnings.warn(
                        'Unable to remember value of variable %s. Original error: \n\n%s: %s' %
                        (variable, cls.__name__, inst.message))

    def trace(self, name, chain=-1):
        """Return the trace of a tallyable object stored in the database.

        :Parameters:
        name : string
          The name of the tallyable object.
        chain : int
          The trace index. Setting `chain=i` will return the trace created by
          the ith call to `sample`.
        """
        if isinstance(name, str):
            return self.db.trace(name, chain)
        elif isinstance(name, Variable):
            return self.db.trace(name.__name__, chain)
        else:
            raise ValueError(
                'Name argument must be string or Variable, got %s.' %
                name)

    def _get_deviance(self):
        return self._sum_deviance()
    deviance = property(_get_deviance)


def check_valid_object_name(sequence):
    """Check that the names of the objects are all different."""
    names = []
    for o in sequence:
        if o.__name__ in names:
            raise ValueError(
                'A tallyable PyMC object called %s already exists. This will cause problems for some database backends.' %
                o.__name__)
        else:
            names.append(o.__name__)
