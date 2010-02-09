import pymc as pm
import numpy 


from pymc.examples import DisasterModel as my_model

S = pm.MCMC(my_model)
S.sample(10000, burn=5000)

# Changed the order of appearance to get the sampling done before this.
pymc_object = S
scores = pm.geweke(pymc_object, first=0.1, last=0.5, intervals=20)
#pm.Matplot.geweke_plot(scores, name='geweke', format='png', suffix='-diagnostic', \
#                path='./', fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}, verbose=1)
#


scores = pm.geweke(S, intervals=20)
pm.Matplot.geweke_plot(scores)

### Replaced alpha by e ###
trace = S.trace('e')[:]
alpha_scores = pm.geweke(trace, intervals=20)
pm.Matplot.geweke_plot(alpha_scores, 'e')


##pm.raftery_lewis(pymc_object, q, r, s=.95, epsilon=.001, verbose=1)


# This has already been done above.
#S = pm.MCMC(my_model)
#S.sample(10000, burn=5000)

pm.raftery_lewis(S, q=0.025, r=0.01)


"""
========================
Raftery-Lewis Diagnostic
========================

937 iterations required (assuming independence) to achieve 0.01 accuracy
with 95 percent probability.

Thinning factor of 1 required to produce a first-order Markov chain.

39 iterations to be discarded at the beginning of the simulation (burn-in).

11380 subsequent iterations required.

Thinning factor of 11 required to produce an independence chain.
"""


pm.utils.coda(pymc_object)

# Check that this really works
#autocorrelation(pymc_object, name, maxlag=100, format='png', suffix='-acf',
#path='./', fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}, verbose=1)


# This has already been done above.
#S = pm.MCMC(my_model)
#S.sample(10000, burn=5000)
pm.Matplot.autocorrelation(S)

# Changed beta for e
pm.Matplot.autocorrelation(S.e)


### SETUP ###
disasters_array =   numpy.array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                   3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                   2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                   1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                   0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                   3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

switchpoint = pm.DiscreteUniform('s', lower=0, upper=110)
early_mean = pm.Exponential('e', beta=1)
late_mean = pm.Exponential('l', beta=1)

@pm.stochastic(observed=True, dtype=int)
def disasters(  value = disasters_array,
                early_mean = early_mean,
                late_mean = late_mean,
                switchpoint = switchpoint):
    """Annual occurences of coal mining disasters."""
    return pm.poisson_like(value[:switchpoint],early_mean) + \
        pm.poisson_like(value[switchpoint:],late_mean)



@pm.deterministic
def disasters_sim(early_mean = early_mean,
                late_mean = late_mean,
                switchpoint = switchpoint):
    """Coal mining disasters sampled from the posterior predictive distribution"""
    return numpy.concatenate( (pm.rpoisson(early_mean, size=switchpoint),
        pm.rpoisson(late_mean, size=len(disasters_array)-switchpoint)))





S = pm.MCMC([disasters_sim, disasters])

x_sim = disasters_sim
x = disasters
#pm.Matplot.gof_plot(x_sim, x, name='x')


#D = pm.diagnostics.discrepancy(observed, simulated, expected)

#pm.Matplot.discrepancy_plot(D, name='D', report_p=True)

