import numpy as np
from pymc import *
from pymc.step_methods.metropolis_hastings import *
import theano
import theano.tensor as T
import pandas
import matplotlib.pylab as plt
#theano.config.mode = 'DebugMode'

model = Model()
with model:

    k = 5
    a = constant(np.array([2, 6., 4, 2, 2]))
    pa = a / T.sum(a)

    p, p_m1 = model.TransformedVar(
        'p', Dirichlet.dist(a, shape=k),
       simplextransform)

    c = Categorical('c', pa)

def run(n=50000):
    if n == "short":
        n = 50
    with model:
        ' Try this with a Metropolis instance, and watch it fail ..'
        step = MetropolisHastings()
        trace = sample(n, step)
    return trace
if __name__ == '__main__':
    tr = run()
    t1 = pandas.Series(tr['p'][:,0])
    t2 = pandas.Series(tr['p'][:,1])
    t3 = pandas.Series(tr['p'][:,2])
    t4 = pandas.Series(tr['p'][:,3])
    t5 = pandas.Series(tr['p'][:,4])
    t6 = pandas.Series(tr['c'])
    df = pandas.DataFrame({'a' : t1,'b' : t2, 'c' : t3, 'd' : t4, 'cat' : t6})
    pandas.scatter_matrix(df)
    plt.show()


