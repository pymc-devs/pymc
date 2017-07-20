from collections import OrderedDict
import warnings

from .arraystep import Competence, ArrayStepShared
from ..vartypes import continuous_types
from ..model import modelcontext, inputvars
import theano.tensor as tt
from ..theanof import tt_rng, make_shared_replacements
import theano
import numpy as np

__all__ = ['SGFS']

EXPERIMENTAL_WARNING = "Warning: Stochastic Gradient based sampling methods are experimental step methods and not yet"\
    " recommended for use in PyMC3!"

def _value_error(cond, str):
    """Throws ValueError if cond is False"""
    if not cond:
        raise ValueError(str)

def _check_minibatches(minibatch_tensors, minibatches):
    _value_error(isinstance(minibatch_tensors, list),
                 'minibatch_tensors must be a list.')

    _value_error(hasattr(minibatches, "__iter__"),
                 'minibatches must be an iterator.')

def prior_dlogp(vars, model, flat_view):
    """Returns the gradient of the prior on the parameters as a vector of size D x 1"""
    terms = tt.concatenate(
        [theano.grad(var.logpt, var).flatten() for var in vars],
        axis=0)
    dlogp = theano.clone(terms, flat_view.replacements, strict=False)

    return dlogp

def elemwise_dlogL(vars, model, flat_view):
    """
    Returns Jacobian of the log likelihood for each training datum wrt vars
    as a matrix of size N x D
    """
    # select one observed random variable 
    obs_var = model.observed_RVs[0]
    # tensor of shape (batch_size,)
    logL = obs_var.logp_elemwiset.sum(axis=tuple(range(1, obs_var.logp_elemwiset.ndim)))
    # calculate fisher information
    terms = []
    for var in vars:
         output, _ =  theano.scan(lambda i, logX=logL, v=var: theano.grad(logX[i], v).flatten(),\
                            sequences=[tt.arange(logL.shape[0])])
         terms.append(output)
    dlogL = theano.clone(tt.concatenate(terms, axis=1), flat_view.replacements, strict=False)
    return dlogL


class BaseStochasticGradient(ArrayStepShared):
    R"""
    BaseStochasticGradient Object
    
    For working with BaseStochasticGradient Object
    we need to supply the probabilistic model 
    (:code:`model`) with the data supplied to observed
    variables of type `GeneratorOp`
    
    Parameters
    -------
    vars : list
        List of variables for sampler
    batch_size`: int
        Batch Size for each step
    total_size : int
        Total size of the training data
    step_size : float
        Step size for the parameter update
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context)
    random_seed : int
        The seed to initialize the Random Stream
    minibatches : iterator
        If the ObservedRV.observed is not a GeneratorOp then this parameter must not be None
    minibatch_tensor : list of tensors
        If the ObservedRV.observed is not a GeneratorOp then this parameter must not be None
        The length of this tensor should be the same as the next(minibatches)

    Notes
    -----
    Defining a BaseStochasticGradient needs
    custom implementation of the following methods:
        - :code: `.mk_training_fn()`
            Returns a theano function which is called for each sampling step
        - :code: `._initialize_values()`
            Returns None it creates class variables which are required for the training fn
    """

    def __init__(self,
                 vars=None,
                 batch_size=None,
                 total_size=None,
                 step_size=1.0,
                 model=None,
                 random_seed=None,
                 minibatches=None,
                 minibatch_tensors=None,
                 **kwargs):
        warnings.warn(EXPERIMENTAL_WARNING)
        
        model = modelcontext(model)
        
        if vars is None:
            vars = model.vars

        vars = inputvars(vars)

        self.model = model
        self.vars = vars
        self.batch_size = batch_size
        self.total_size = total_size
        _value_error(
            total_size != None or batch_size != None,
            'total_size and batch_size of training data have to be specified')
        self.expected_iter = int(total_size / batch_size)

        # set random stream
        self.random = None
        if random_seed is None:
            self.random = tt_rng()
        else:
            self.random = tt_rng(random_seed)

        self.step_size = step_size
        shared = make_shared_replacements(vars, model)
        self.q_size = int(sum(v.dsize for v in self.vars))
        
        flat_view = model.flatten(vars)
        self.inarray = [flat_view.input]
        self.updates = OrderedDict()
        self.dlog_prior = prior_dlogp(vars, model, flat_view)
        self.dlogp_elemwise = elemwise_dlogL(vars, model, flat_view)

        if minibatch_tensors != None:
            _check_minibatches(minibatch_tensors, minibatches)
            self.minibatches = minibatches
            
            # Replace input shared variables with tensors
            def is_shared(t):
                return isinstance(t, theano.compile.sharedvalue.SharedVariable)
            tensors = [(t.type() if is_shared(t) else t) for t in minibatch_tensors]
            updates = OrderedDict(
                {t: t_ for t, t_ in zip(minibatch_tensors, tensors) if is_shared(t)}
            )
            self.minibatch_tensors = tensors
            self.inarray += self.minibatch_tensors
            self.updates.update(updates)

        self._initialize_values()
        super(BaseStochasticGradient, self).__init__(vars, shared)

    def _initialize_values(self):
        """Initializes the parameters for the stochastic gradient minibatch
        algorithm"""
        raise NotImplementedError
    
    def mk_training_fn(self):
        raise NotImplementedError

    def training_complete(self):
        """Returns boolean if astep has been called expected iter number of times"""
        return self.expected_iter == self.t

    def astep(self, q0):
        """Perform a single update in the stochastic gradient method.
        
        Returns new shared values and values sampled
        The size and ordering of q0 and q must be the same
        Parameters
        -------
        q0: list
            List of shared values and values sampled from last estimate

        Returns
        -------
        q
        """
        if hasattr(self, 'minibatch_tensors'):
            return q0 + self.training_fn(q0, *next(self.minibatches))
        else:
            return q0 + self.training_fn(q0)


class SGFS(BaseStochasticGradient):
    R"""
    StochasticGradientFisherScoring
    
    Parameters
    -----
    vars : list
        model variables
    B : np.array
        the pre-conditioner matrix for the fisher scoring step

    References
    -----
    -   Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring
        Implements Algorithm 1 from the publication http://people.ee.duke.edu/%7Elcarin/782.pdf
    """

    def __init__(self, vars=None, B=None, **kwargs):
        """
        Parameters
        ----------
        vars : list 
            Theano variables, default continuous vars
        B : np.array
            Symmetric positive Semi-definite Matrix
        kwargs: passed to BaseHMC
        """
        self.B = B
        super(SGFS, self).__init__(vars, **kwargs)

    def _initialize_values(self):
        # Init avg_I
        self.avg_I = theano.shared(np.zeros((self.q_size, self.q_size)), name='avg_I')
        self.t = theano.shared(1, name='t')
        # 2. Set gamma
        self.gamma = (self.batch_size + self.total_size) / (self.total_size)
        
        self.training_fn = self.mk_training_fn()

    def mk_training_fn(self):

        n = self.batch_size
        N = self.total_size
        q_size = self.q_size
        B = self.B
        gamma = self.gamma
        avg_I = self.avg_I
        t = self.t
        updates = self.updates
        step_size = self.step_size
        random = self.random
        inarray = self.inarray
        gt, dlog_prior = self.dlogp_elemwise, self.dlog_prior

        # 5. Calculate mean dlogp
        avg_gt = gt.mean(axis=0)

        # 6. Calculate approximate Fisher Score
        gt_diff = (gt - avg_gt)

        V = (1. / (n - 1)) * tt.dot(gt_diff.T, gt_diff)

        # 7. Update moving average
        I_t = (1. - 1. / t) * avg_I + (1. / t) * V

        if B is None:
            # if B is not specified 
            # B \propto I_t as given in
            # http://www.ics.uci.edu/~welling/publications/papers/SGFS_v10_final.pdf
            # after iterating over the data few times to get a good approximation of I_N
            B = tt.switch(t <= int(N/n)*50, tt.eye(q_size), gamma * I_t)

        # 8. Noise Term
        # The noise term is sampled from a normal distribution
        # of mean 0 and std_dev = sqrt(4B/step_size)
        # In order to generate the noise term, a standard
        # normal dist. is scaled with 2B_ch/sqrt(step_size)
        # where B_ch is cholesky decomposition of B
        # i.e. B = dot(B_ch, B_ch^T)
        B_ch = tt.slinalg.cholesky(B)
        noise_term = tt.dot((2.*B_ch)/tt.sqrt(step_size), \
                random.normal((q_size,), dtype=theano.config.floatX))
        # 9.
        # Inv. Fisher Cov. Matrix
        cov_mat = (gamma * I_t * N) + ((4. / step_size) * B)
        inv_cov_mat = tt.nlinalg.matrix_inverse(cov_mat)
        # Noise Coefficient
        noise_coeff = (dlog_prior + (N * avg_gt) + noise_term)
        dq = 2 * tt.dot(inv_cov_mat, noise_coeff)

        updates.update({avg_I: I_t, t: t + 1})

        f = theano.function(
            outputs=dq,
            inputs=inarray,
            updates=updates,
            allow_input_downcast=True)

        return f

    @staticmethod
    def competence(var):
        if var.dtype in continuous_types:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE
