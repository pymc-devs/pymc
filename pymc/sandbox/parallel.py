import subprocess
from IPython.kernel import client
    

"""
It seems to work, but the real challenge is to get the results back. 
One solution may be to create a parallel database backend. The backend
is instantiated in Parallel, communicates with each process, and serves 
as a middle man with the real backend. Each time the tally method is called, 
it calls the real backend tally with the correct chain. This requires setting 
an optional argument for tally to each backend. 
def tally(self, index, chain=-1):

The sample method is Parallel must initialize the chains
"""


class Parallel:
    """
    Parallel manages multiple MCMC loops. It is initialized with:

    A = Parallel(prob_def, dbase=None, chains=1, proc=1)

    Arguments
    
        prob_def: class, module or dictionary containing nodes and 
        StepMethods)
        
        dbase: Database backend used to tally the samples. 
        Implemented backends: None, hdf5.
        
        proc: Number of processes (generally the number of available CPUs.)
        
    Externally-accessible attributes:

        dtrms:          All extant Deterministics.

        stochs:         All extant Stochastics with isdata = False.

        data:               All extant Stochastics with isdata = True.

        nodes:               All extant Stochastics and Deterministics.

        step_methods:   All extant StepMethods.

    Externally-accessible methods:

        sample(iter,burn,thin): At each MCMC iteration, calls each step_method's step() method.
                                Tallies Stochastics and Deterministics as appropriate.

        trace(stoch, burn, thin, slice): Return the trace of stoch, 
        sliced according to slice or burn and thin arguments.

        remember(trace_index): Return the entire model to the tallied state indexed by trace_index.

        DAG: Draw the model as a directed acyclic graph.

        All the plotting functions can probably go on the base namespace and take Stochastics as
        arguments.

    See also StepMethod, OneAtATimeMetropolis, Node, Stochastic, Deterministic, and weight.
    """
    def __init__(self, input, dbase='ram', proc=2):
        try:
            mec = client.MultiEngineClient()
        except:
            p = subprocess.Popen('ipcluster -n %d'%proc, shell=True)
            p.wait()
            mec = client.MultiEngineClient()
        
        # Check everything is alright.
        nproc = len(mec.get_ids())
        
        # Import the individual models in each process
        #mec.pushModule(input)
        
        try:
            mec.execute('import %s as input'%input.__name__)
        except:
            mec.execute( 'import site' )
            mec.execute( 'site.addsitedir( ' + `os.getcwd()` + ' )' )
            mec.execute( 'import %s as input; reload(input)'%input.__name__)
        
        # Instantiate Sampler instances in each process
        mec.execute('from pymc import MCMC')
        #mec.execute('from pymc.database.parallel import Database')
        #for i in range(nproc):
        #    mec.execute(i, 'db = Database(%d)'%i)
        mec.execute('S = MCMC(input)')
        
        self.mec = mec
        
    def sample(self, iter, burn=0, thin=1, tune_interval=100):
        # Set the random initial seeds
        self.mec.execute('S.seed()')
        
        # Run the chains on each process
        self.mec.execute('S.sample(%(iter)i, %(burn)i, %(thin)i, %(tune_interval)i)'%vars())
        
        # Merge the traces
         
        
    

if __name__ == '__main__':
    from pymc.examples import DisasterModel
    P = Parallel(DisasterModel, 'ram')
    P.sample(1000,500,1)
    #P.mec.killAll(controller=True)
