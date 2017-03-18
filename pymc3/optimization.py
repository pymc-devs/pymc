import numpy as np
import theano
import tqdm
from pymc3.backends.base import BaseTrace, MultiTrace
from pymc3.variational import Histogram, adam


class Optimizer(object):
    def __init__(self, approx, loss, wrt, more_replacements=None, model=None, optimizer=None):
        if optimizer is None:
            optimizer = adam
        self.optimizer = optimizer
        self.wrt = wrt
        if isinstance(approx, (BaseTrace, MultiTrace)):
            approx = Histogram(approx, model=model)
        self.approx = approx
        self.loss = approx.apply_replacements(loss, more_replacements=more_replacements)
        updates = self.optimizer(self.loss, self.wrt)
        self.step_function = theano.function([], self.loss, updates=updates)
        self.hist = np.asarray(())

    def refresh(self):
        updates = self.optimizer(self.loss, self.wrt)
        self.step_function = theano.function([], self.loss, updates=updates)
        self.hist = np.asarray(())

    def fit(self, n=5000):
        progress = tqdm.trange(n)
        scores = np.empty(n)
        scores[:] = np.nan
        with progress:
            for i in progress:
                scores[i] = self.step_function()
                if i % (n//1000) == 0:
                    progress.set_description(
                        'E_q[Loss] = %.4f' % scores[max(0, i - n//50):i+1].mean()
                    )
        return self.wrt
