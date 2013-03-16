import matplotlib.pyplot as p
import numpy as np
from scipy.stats import kde 

__all__ = ['traceplot']

def traceplot(trace, vars=None): 
    if vars is None:
        vars = trace.samples.keys()

    n = len(vars)
    f, ax = p.subplots(2, n, squeeze = False)

    for i,v in enumerate(vars):
        d = np.squeeze(trace[v])

        kdeplot(ax[0,i], d)
        ax[1,i].plot(d)

    return f 

def kdeplot(ax, data):
    data = np.atleast_2d(data.T).T
    for i in range(data.shape[1]):
        d = data[:,i]
        density = kde.gaussian_kde(d) 
        l = np.min(d)
        u = np.max(d)
        x = np.linspace(0,1,100)*(u-l)+l

        ax.plot(x,density(x))
    




