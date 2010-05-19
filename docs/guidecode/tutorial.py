# Code from tutorial.tex

"""This is just a copy of examples/DisasterModel.py"""
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np



disasters_array =   np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])



s = DiscreteUniform('s', lower=0, upper=110, doc='Switchpoint[year]')



e = Exponential('e', beta=1)
l = Exponential('l', beta=1)



@deterministic(plot=False)
def r(s=s, e=e, l=l):
    """ Concatenate Poisson means """
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out



D = Poisson('D', mu=r, value=disasters_array, observed=True)


from pymc.examples import DisasterModel
DisasterModel.s.parents
#{'lower': 0, 'upper': 110}


DisasterModel.D.parents
#{'mu': <pymc.PyMCObjects.Deterministic 'r' at 0x3e51a70>}


DisasterModel.r.children
#set([<pymc.distributions.Poisson 'D' at 0x3e51290>])


DisasterModel.D.value
#array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1,
#      4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3,
#      0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0,
#      0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2,
#      0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

DisasterModel.s.value
#44

DisasterModel.e.value
#0.33464706250079584

DisasterModel.l.value
#2.6491936762267811



DisasterModel.r.value
#   array([ 0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  0.33464706,
#           0.33464706,  0.33464706,  0.33464706,  0.33464706,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368,
#           2.64919368,  2.64919368,  2.64919368,  2.64919368,  2.64919368])



DisasterModel.s.logp
#-4.7095302013123339

DisasterModel.D.logp
#-1080.5149888046033

DisasterModel.e.logp
#-0.33464706250079584

DisasterModel.l.logp
#-2.6491936762267811



@deterministic(plot=False)
def r(s=s, e=e, l=l):
    """ Concatenate Poisson means """
    out = np.empty(len(disasters_array))
    out[:s] = e
    out[s:] = l
    return out



from pymc.examples import DisasterModel
from pymc import MCMC
M = MCMC(DisasterModel)



M.isample(iter=10000, burn=1000, thin=10)



M.trace('s')[:]
#array([41, 40, 40, ..., 43, 44, 44])



from pylab import hist, show
hist(M.trace('l')[:])
#(array([   8,   52,  565, 1624, 2563, 2105, 1292,  488,  258,   45]),
#array([ 0.52721865,  0.60788251,  0.68854637,  0.76921023,  0.84987409,
#   0.93053795,  1.01120181,  1.09186567,  1.17252953,  1.25319339]),
#<a list of 10 Patch objects>)
show()



from pymc.Matplot import plot
plot(M)



x = np.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
2, 2, 3, 4, 2, 1, 3, None, 2, 1, 1, 1, 1, 3, 0, 0,
1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
3, 3, 1, None, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])



masked_data = np.ma.masked_equal(x, value=None)
masked_data
#masked_array(data = [4 5 4 0 1 4 3 4 0 6 3 3 4 0 2 6 3 3 5 4 5 3 1 4 4 1 5 5 3
#4 2 5 2 2 3 4 2 1 3 -- 2 1 1 1 1 3 0 0 1 0 1 1 0 0 3 1 0 3 2 2 0 1 1 1 0 1 0
#1 0 0 0 2 1 0 0 0 1 1 0 2 3 3 1 -- 2 1 1 1 1 2 4 2 0 0 1 4 0 0 0 1 0 0 0 0 0 1
#0 0 1 0 1],
#mask = [False False False False False False False False False False False False
#False False False False False False False False False False False False
#False False False False False False False False False False False False
#False False False  True False False False False False False False False
#False False False False False False False False False False False False
#False False False False False False False False False False False False
#False False False False False False False False False False False  True
#False False False False False False False False False False False False
#False False False False False False False False False False False False
#False False False],
#fill_value=?)




from pymc import Impute
D = Impute('D', Poisson, masked_data, mu=r)
D
#[<pymc.distributions.Poisson 'D[0]' at 0x4ba42d0>,
#<pymc.distributions.Poisson 'D[1]' at 0x4ba4330>,
#<pymc.distributions.Poisson 'D[2]' at 0x4ba44d0>,
#<pymc.distributions.Poisson 'D[3]' at 0x4ba45f0>,
#...
#<pymc.distributions.Poisson 'D[110]' at 0x4ba46d0>]



# Switchpoint
s = DiscreteUniform('s', lower=0, upper=110)
# Early mean
e = Exponential('e', beta=1)
# Late mean
l = Exponential('l', beta=1)

@deterministic(plot=False)
def r(s=s, e=e, l=l):
    """Allocate appropriate mean to time series"""
    out = np.empty(len(disasters_array))
    # Early mean prior to switchpoint
    out[:s] = e
    # Late mean following switchpoint
    out[s:] = l
    return out

# Where the value of x is None, the value is taken as missing.
D = Impute('D', Poisson, x, mu=r)



M.step_method_dict[DisasterModel.s]
#[<pymc.StepMethods.DiscreteMetropolis object at 0x3e8cb50>]

M.step_method_dict[DisasterModel.e]
#[<pymc.StepMethods.Metropolis object at 0x3e8cbb0>]

M.step_method_dict[DisasterModel.l]
#[<pymc.StepMethods.Metropolis object at 0x3e8ccb0>]



from pymc import Metropolis
M.use_step_method(Metropolis, DisasterModel.l, proposal_sd=2.)

