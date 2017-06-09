"""Text file trace backend modified to work efficiently with SMC

Store sampling values as CSV files.

File format
-----------

Sampling values for each chain are saved in a separate file (under a
directory specified by the `name` argument).  The rows correspond to
sampling iterations.  The column names consist of variable names and
index labels.  For example, the heading

  x,y__0_0,y__0_1,y__1_0,y__1_1,y__2_0,y__2_1

represents two variables, x and y, where x is a scalar and y has a
shape of (3, 2).
"""
from glob import glob
import itertools
import multiprocessing
import pickle
import os
import shutil
from six.moves import map, zip

import pandas as pd
import pymc3 as pm

from ..model import modelcontext
from ..backends import base, ndarray
from . import tracetab as ttab
from ..blocking import DictToArrayBijection, ArrayOrdering,\
                       ListArrayOrdering, ListToArrayBijection
from ..step_methods.arraystep import BlockedStep


def paripool(function, work, nprocs=None, chunksize=1):
    """Initialise a pool of workers and execute a function in parallel by
    forking the process. Does forking once during initialisation.

    Parameters
    ----------
    function : function
        python function to be executed in parallel
    work : list
        of iterables that are to be looped over/ executed in parallel usually
        these objects are different for each task.
    nprocs : int
        number of processors to be used in paralell process
    chunksize : int
        number of work packages to throw at workers in each instance
    """
    if nprocs is None:
        nprocs = multiprocessing.cpu_count()

    if nprocs == 1:
        for work_item in work:
            yield function(work_item)
    else:
        try:
            pool = multiprocessing.Pool(processes=nprocs)
            yield pool.map(function, work, chunksize=chunksize)
        finally:
            pool.terminate()


class ArrayStepSharedLLK(BlockedStep):
    """Modified ArrayStepShared To handle returned larger point including the likelihood values.
    Takes additionally a list of output vars including the likelihoods.

    Parameters
    ----------

    vars : list
        variables to be sampled
    out_vars : list
        variables to be stored in the traces
    shared : dict
        theano variable -> shared variables
    blocked : boolen
        (default True)
    """
    def __init__(self, vars, out_vars, shared, blocked=True):
        self.vars = vars
        self.ordering = ArrayOrdering(vars)
        self.lordering = ListArrayOrdering(out_vars, intype='tensor')
        lpoint = [var.tag.test_value for var in out_vars]
        self.shared = {var.name: shared for var, shared in shared.items()}
        self.blocked = blocked
        self.bij = DictToArrayBijection(self.ordering, self.population[0])
        self.lij = ListToArrayBijection(self.lordering, lpoint)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, point):
        for var, share in self.shared.items():
            share.container.storage[0] = point[var]

        apoint, alist = self.astep(self.bij.map(point))
        return self.bij.rmap(apoint), alist


class BaseSMCTrace(object):
    """Base SMC trace object

    Parameters
    ----------

    name : str
        Name of backend
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    def __init__(self, name, model=None, vars=None):
        self.name = name
        model = modelcontext(model)
        self.model = model

        if vars is None:
            vars = model.unobserved_RVs

        self.vars = vars
        self.varnames = [var.name for var in vars]

        #  Get variable shapes. Most backends will need this information.

        self.var_shapes_list = [var.tag.test_value.shape for var in vars]
        self.var_dtypes_list = [var.tag.test_value.dtype for var in vars]

        self.var_shapes = dict(zip(self.varnames, self.var_shapes_list))
        self.var_dtypes = dict(zip(self.varnames, self.var_dtypes_list))

        self.chain = None

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)
        try:
            return self.point(int(idx))
        except (ValueError, TypeError):  # Passed variable or variable name.
            raise ValueError('Can only index with slice or integer')

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class TextStage(object):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.project_dir = os.path.dirname(base_dir)
        self.mode = os.path.basename(base_dir)
        if not os.path.isdir(base_dir):
            os.mkdir(self.base_dir)

    def stage_path(self, stage):
        return os.path.join(self.base_dir, 'stage_{}'.format(stage))

    def stage_number(self, stage_path):
        """Inverse function of TextStage.path"""
        return int(os.path.basename(stage_path).split('_')[-1])

    def highest_sampled_stage(self):
        """Return stage number of stage that has been sampled before the final stage.

        Returns
        -------
        stage number : int
        """
        return max(self.stage_number(s) for s in glob(self.stage_path('*')))

    def atmip_path(self, stage_number):
        """Consistent naming for atmip params."""
        return os.path.join(self.stage_path(stage_number), 'atmip.params.pkl')

    def load_atmip_params(self, stage_number, model):
        """Load saved parameters from last sampled ATMIP stage.

        Parameters
        ----------
        stage number : int
            of stage number or -1 for last stage
        """
        if stage_number == -1:
            prev = self.highest_sampled_stage()
        else:
            prev = stage_number - 1
        pm._log.info('Loading parameters from completed stage {}'.format(prev))

        with model:
            with open(self.atmip_path(prev), 'rb') as buff:
                step = pickle.load(buff)

        # update step stage to current stage
        step.stage = stage_number
        return step

    def dump_atmip_params(self, step):
        """Save atmip params to file."""
        with open(self.atmip_path(step.stage), 'wb') as buff:
            pickle.dump(step, buff, protocol=pickle.HIGHEST_PROTOCOL)

    def clean_directory(self, stage, chains, rm_flag):
        """Optionally remove directory for the stage.  Does nothing if rm_flag is False."""
        stage_path = self.stage_path(stage)
        if rm_flag:
            if os.path.exists(stage_path):
                pm._log.info('Removing previous sampling results ... %s' % stage_path)
                shutil.rmtree(stage_path)
            chains = None
        elif not os.path.exists(stage_path):
            chains = None
        return chains

    def load_multitrace(self, stage, model=None):
        """Load TextChain database.

        Parameters
        ----------
        name : str
            Name of directory with files (one per chain)
        model : Model
            If None, the model is taken from the `with` context.

        Returns
        -------

        A :class:`pymc3.backend.base.MultiTrace` instance
        """
        dirname = self.stage_path(stage)
        files = glob(os.path.join(dirname, 'chain_*.csv'))
        straces = []
        for f in files:
            chain = int(os.path.basename(f).split('_')[-1].split('.')[0])
            strace = TextChain(dirname, model=model)
            strace.chain = chain
            strace.filename = f
            straces.append(strace)
        return base.MultiTrace(straces)

    def check_multitrace(self, mtrace, draws, n_chains):
        """Check multitrace for incomplete sampling and return indexes from chains
        that need to be resampled.

        Parameters
        ----------
        mtrace : :class:`pymc3.backend.base.MultiTrace`
            Multitrace object containing the sampling traces
        draws : int
            Number of steps (i.e. chain length for each Markov Chain)
        n_chains : int
            Number of Markov Chains

        Returns
        -------
        list of indexes for chains that need to be resampled
        """
        not_sampled_idx = []
        for chain in range(n_chains):
            if chain in mtrace.chains:
                if len(mtrace._straces[chain]) != draws:
                    pm._log.warn('Trace number %i incomplete' % chain)
                    mtrace._straces[chain].corrupted_flag = True
            else:
                not_sampled_idx.append(chain)

        flag_bool = [mtrace._straces[chain].corrupted_flag for chain in mtrace.chains]
        corrupted_idx = [i for i, x in enumerate(flag_bool) if x]
        return corrupted_idx + not_sampled_idx

    def recover_existing_results(self, stage, draws, step, n_jobs, model=None):
        stage_path = self.stage_path(stage)
        if os.path.exists(stage_path):
            # load incomplete stage results
            pm._log.info('Reloading existing results ...')
            mtrace = self.load_multitrace(stage, model=model)
            if len(mtrace.chains) > 0:
                # continue sampling if traces exist
                pm._log.info('Checking for corrupted files ...')
                return self.check_multitrace(mtrace, draws=draws, n_chains=step.n_chains)
        pm._log.info('Init new trace!')
        return None


class TextChain(BaseSMCTrace):
    """Text trace object

    Parameters
    ----------

    name : str
        Name of directory to store text files
    model : Model
        If None, the model is taken from the `with` context.
    vars : list of variables
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    """

    def __init__(self, name, model=None, vars=None):
        if not os.path.exists(name):
            os.mkdir(name)
        super(TextChain, self).__init__(name, model, vars)

        self.flat_names = {v: ttab.create_flat_names(v, shape)
                           for v, shape in self.var_shapes.items()}

        self.filename = None
        self.df = None
        self.corrupted_flag = False

    def setup(self, draws, chain):
        """Perform chain-specific setup.

        Parameters
        ----------
        draws : int
            Expected number of draws
        chain : int
            Chain number
        """
        self.chain = chain
        self.filename = os.path.join(self.name, 'chain_{}.csv'.format(chain))

        cnames = [fv for v in self.varnames for fv in self.flat_names[v]]

        if os.path.exists(self.filename):
            os.remove(self.filename)

        with open(self.filename, 'w') as fh:
            fh.write(','.join(cnames) + '\n')

    def record(self, lpoint):
        """Record results of a sampling iteration.

        Parameters
        ----------
        lpoint : List of variable values
            Values mapped to variable names
        """
        columns = itertools.chain.from_iterable(map(str, value.ravel()) for value in lpoint)
        with open(self.filename, 'a') as fh:
            fh.write(','.join(columns) + '\n')

    def _load_df(self):
        if self.df is None:
            try:
                self.df = pd.read_csv(self.filename)
            except pd.parser.EmptyDataError:
                pm._log.warn('Trace %s is empty and needs to be resampled!' % self.filename)
                os.remove(self.filename)
                self.corrupted_flag = True
            except pd.io.common.CParserError:
                pm._log.warn('Trace %s has wrong size!' % self.filename)
                self.corrupted_flag = True
                os.remove(self.filename)

    def __len__(self):
        if self.filename is None:
            return 0

        self._load_df()

        if self.df is None:
            return 0
        else:
            return self.df.shape[0]

    def get_values(self, varname, burn=0, thin=1):
        """Get values from trace.

        Parameters
        ----------
        varname : str
            Variable name for which values are to be retrieved.
        burn : int
            Burn-in samples from trace. This is the number of samples to be
            thrown out from the start of the trace
        thin : int
            Nuber of thinning samples. Throw out every 'thin' sample of the
            trace.

        Returns
        -------

        :class:`numpy.array`
        """
        self._load_df()
        var_df = self.df[self.flat_names[varname]]
        shape = (self.df.shape[0],) + self.var_shapes[varname]
        vals = var_df.values.ravel().reshape(shape)
        return vals[burn::thin]

    def _slice(self, idx):
        if idx.stop is not None:
            raise ValueError('Stop value in slice not supported.')
        return ndarray._slice_as_ndarray(self, idx)

    def point(self, idx):
        """Get point of current chain with variables names as keys.

        Parameters
        ----------
        idx : int
            Index of the nth step of the chain

        Returns
        -------
        dictionary of point values
        """
        idx = int(idx)
        self._load_df()
        pt = {}
        for varname in self.varnames:
            vals = self.df[self.flat_names[varname]].iloc[idx]
            pt[varname] = vals.reshape(self.var_shapes[varname])
        return pt
