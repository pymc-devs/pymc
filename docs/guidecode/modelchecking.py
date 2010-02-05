import pymc as pm
import numpy 

### SETUP ###
from pymc.examples import DisasterModel
pymc_object = pm.MCMC(DisasterModel)
pymc_object.sample(5000)

scores = pm.geweke(pymc_object, first=0.1, last=0.5, intervals=20)



pm.Matplot.geweke_plot(scores, name='geweke', format='png', suffix='-diagnostic', \
                path='./', fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}, verbose=1)


from pymc.examples import DisasterModel as my_model

S = pm.MCMC(my_model)
S.sample(10000, burn=5000)

scores = pm.geweke(S, intervals=20)
pm.Matplot.geweke_plot(scores)

trace = S.alpha.trace()
alpha_scores = pm.geweke(trace, intervals=20)
pm.Matplot.geweke_plot(alpha_scores, 'alpha')


raftery_lewis(pymc_object, q, r, s=.95, epsilon=.001, verbose=1)



S = pm.MCMC(my_model)
S.sample(10000, burn=5000)



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


autocorrelation(pymc_object, name, maxlag=100, format='png', suffix='-acf',
path='./', fontmap = {1:10, 2:8, 3:6, 4:5, 5:4}, verbose=1)



S = pm.MCMC(my_model)
S.sample(10000, burn=5000)
pm.Matplot.autocorrelation(S)



pm.Matplot.autocorrelation(S.beta)


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
    return concatenate( (pm.rpoisson(early_mean, size=switchpoint),
        pm.rpoisson(late_mean, size=n-switchpoint)))



pm.Matplot.gof_plot(x_sim, x, name='x')



D = pm.diagnostics.discrepancy(observed, simulated, expected)



pm.Matplot.discrepancy_plot(D, name='D', report_p=True)

