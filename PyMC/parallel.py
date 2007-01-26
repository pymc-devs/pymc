import subprocess
import ipython1.kernel.api as kernel
import proposition5
"""The devs of ipython are refactoring the chainsaw branch. We should probably 
wait awhile before commiting to much time on this, until their API is stable."""


class Parallel(Model):
    """
    Parallel manages multiple MCMC loops. It is initialized with:

    A = Parallel(prob_def, dbase=None, chains=1, proc=1)

    Arguments
    
        prob_def: class, module or dictionary containing PyMC objects and 
        SamplingMethods)
        
        dbase: Database backend used to tally the samples. 
        Implemented backends: None, hdf5.

        chains: Number of loops with different initial conditions.
        
        proc: Number of processes (generally the number of available CPUs.)
        
    Externally-accessible attributes:

        nodes:          All extant Nodes.

        parameters:         All extant Parameters with isdata = False.

        data:               All extant Parameters with isdata = True.

        pymc_objects:               All extant Parameters and Nodes.

        sampling_methods:   All extant SamplingMethods.

    Externally-accessible methods:

        sample(iter,burn,thin): At each MCMC iteration, calls each sampling_method's step() method.
                                Tallies Parameters and Nodes as appropriate.

        trace(parameter, burn, thin, slice): Return the trace of parameter, 
        sliced according to slice or burn and thin arguments.

        remember(trace_index): Return the entire model to the tallied state indexed by trace_index.

        DAG: Draw the model as a directed acyclic graph.

        All the plotting functions can probably go on the base namespace and take Parameters as
        arguments.

    See also SamplingMethod, OneAtATimeMetropolis, PyMCBase, Parameter, Node, and weight.
    """
    def __init__(self, input, dbase=None, chains=1, proc=1):
        try:
            rc = kernel.RemoteController(('127.0.0.1',10105))
        except:
            p = subprocess.Popen('ipcluster -n %d &'%proc, shell=True)
            p.wait()
            rc = kernel.RemoteController(('127.0.0.1',10105))
        
        # Check everything is alright.
        rc.getIDs()
        
        # Push the individual models in each process
        # Works with ipython1 since jan 18. 2007. Buggy before that. 
        rc.pushModule(input)
        # still buggy
        rc.executeAll('import input') # input.__name__?
        # Initiate Model instances in each process
        rc.pushModule(proposition5)
        
        # Set the random initial seeds
        
        # Run the chains on each process
        
        # Merge the traces
         
        rc.killAll(controller=True)
    
