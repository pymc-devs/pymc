import tqdm
import numpy as np
import theano
from theano.configparser import change_flags
import pymc3 as pm


class Optimizer(object):
    """
    Optimization with posterior replacements
    Parameters
    ----------
    approx : Approximation
    loss : scalar
    wrt : shared params
    more_replacements : other replacements in the graph
    optimizer : callable that returns updates, pm.adam by default
    """
    @change_flags(compute_test_value='off')
    def __init__(self, approx, loss, wrt, more_replacements=None, optimizer=None):
        if optimizer is None:
            optimizer = pm.adam
        self.optimizer = optimizer
        self.wrt = wrt
        self.approx = approx
        self.loss = approx.apply_replacements(loss, more_replacements=more_replacements)
        updates = self.optimizer(self.loss, self.wrt)
        self.step_function = theano.function([], self.loss, updates=updates)
        self.hist = np.asarray(())

    @change_flags(compute_test_value='off')
    def refresh(self, kwargs=None):
        """
        Recompile step_function and reset updates

        Parameters
        ----------
        kwargs : kwargs for theano.function
        """
        updates = self.optimizer(self.loss, self.wrt)
        self.step_function = theano.function([], self.loss, updates=updates, **kwargs)
        self.hist = np.asarray(())

    def fit(self, n=5000, callbacks=()):
        """
        Perform optimization steps
        Parameters
        ----------
        n : int
            number of iterations
        callbacks : list[callable]
            list of callables with following signature
            f(Approximation, loss_history, i) -> None
        """
        progress = tqdm.trange(n)
        scores = np.empty(n)
        scores[:] = np.nan
        i = 0
        try:
            for i in progress:
                scores[i] = self.step_function()
                for callback in callbacks:
                    callback(self.approx, scores[:i + 1], i)
                if i % ((n+1000)//1000) == 0:
                    progress.set_description(
                        'E_q[Loss] = %.4f' % scores[max(0, i - n//50):i+1].mean()
                    )
        except KeyboardInterrupt:
            pass
        finally:
            self.hist = np.concatenate((self.hist, scores[:i]))
            progress.close()
