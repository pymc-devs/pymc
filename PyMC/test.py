from scipy import *
from pylab import *
from numpy import *
from base_classes import *
from parameter_library import *
from sampler_library import *
from TimeSeries import *
from Matplot import *

"""
Declare coefficients of the curves, give them Normal priors
"""
Rick_1 = Normal("first Ricker coefficient",.1,scale = .1)
Rick_2 = Normal("second Ricker Ricker coefficient",.2,scale = .1)

BH_1 = Normal("first Beverton-Holt coefficient",.1,scale = .1)
BH_2 = Normal("second Beverton-Holt coefficient",.1,scale = .1)

"""
Declare observation precision, give it a Gamma prior
"""
tau = Gamma("Observation precision",100.0,scale=50)

"""
Declare prior parameters as Constant
"""
mu_Rick_1 = Constant("Mean of first Ricker coefficient",1.0)
tau_Rick_1 = Constant("Precision of first Ricker coefficient",.01)

mu_Rick_2 = Constant("Mean of second Ricker Ricker coefficient",1.0)
tau_Rick_2 = Constant("Precision of second Ricker Ricker coefficient",.01)

mu_BH_1 = Constant("Mean of first Beverton-Holt coefficient",1.0)
tau_BH_1 = Constant("Precision of first Beverton-Holt coefficient",.01)

mu_BH_2 = Constant("Mean of second Beverton-Holt coefficient",1.0)
tau_BH_2 = Constant("Precision of second Beverton-Holt coefficient",.01)

alpha = Constant("alpha of observation precision",50.0)
beta = Constant("beta of observation precision",2.0)

"""
Declare the deterministic functions mapping observation locations to expectation of observation
"""
# Ricker( c1, c2, x ) = c1 x exp(-c2 x)
Ricker = DeterministicFunction("Ricker curve",lambda c1,c2,x: c1 * x *exp(-1.0 * c2 * x),("c1","c2","x"))
# BH( c1, c2, x ) = c1 x / (1 + c2 x)
BH = DeterministicFunction("Beverton-Holt curve",lambda c1,c2,x: c1 * x / (1.0 + c2 * x),("c1","c2","x"))


"""
Declare observation locations and make simulated data
"""
x_axis = resize(arange(0.0,10.0),-1)
x_Rick = Constant("Locations of observations",x_axis)
x_BH = Constant("Locations of observations",x_axis)
data_Rick = Normal("Observations of Ricker curve",.1 * x_axis *exp(- .2 * x_axis) + .1 * randn(9), observed = True)
data_BH = Normal("Observations of Beverton-Holt curve",.1 * x_axis / (1.0 + .1 * x_axis) + .1 * randn(9), observed = True)


"""
Establish dependencies. The code would be shorter if parents were declared first and passed into their
childrens' constructors, but it's a bit safer and easier to see what's going on this way.
"""
#Ricker_1 ~ N( mu_Rick_1, tau_Rick_1 )
Rick_1.add_parent(mu_Rick_1,"mu")
Rick_1.add_parent(tau_Rick_1,"tau")

#Ricker_2 ~ N( mu_Rick_2, tau_Rick_2 )
Rick_2.add_parent(mu_Rick_2,"mu")
Rick_2.add_parent(tau_Rick_2,"tau")

#BH_1 ~ N( mu_BH_1, tau_BH_1 )
BH_1.add_parent(mu_BH_1,"mu")
BH_1.add_parent(tau_BH_1,"tau")

#BH_2 ~ N( mu_BH_2, tau_BH_2 )
BH_2.add_parent(mu_BH_2,"mu")
BH_2.add_parent(tau_BH_2,"tau")

#tau ~ Gamma( alpha, beta )
tau.add_parent(alpha,"alpha")
tau.add_parent(beta,"beta")

#Feed coefficients and x-values into Ricker curve
Ricker.add_parent(Rick_1,"c1")
Ricker.add_parent(Rick_2,"c2")
Ricker.add_parent(x_Rick,"x")

#Feed coefficients and x-values into Beverton-Holt curve
BH.add_parent(BH_1,"c1")
BH.add_parent(BH_2,"c2")
BH.add_parent(x_BH,"x")

#data_Rick_i ~ind N( Ricker( Rick_1, Rick_2, x_i ), tau )
data_Rick.add_parent(Ricker,"mu")
data_Rick.add_parent(tau,"tau")

#data_BH_i ~ind N( BH( BH_1, BH_2, x_i ), tau )
data_BH.add_parent(BH,"mu")
data_BH.add_parent(tau,"tau")

thin = 5
trace_length = 2000

Rick_1.init_trace(trace_length)
Rick_2.init_trace(trace_length)
BH_1.init_trace(trace_length)
BH_2.init_trace(trace_length)
tau.init_trace(trace_length)


for i in xrange(0,trace_length*thin):
	if i%1000 == 0:
		print "iteration ",i," of ",trace_length * thin

	"""
	Step all parameters and tally. This could be handled by a Sampler, 
	(assuming I haven't broken Sampler, which is unlikely), 
	but it's easier to see what's going on this way.
	"""

	Rick_1.metropolis_step()
	Rick_2.metropolis_step()
	BH_1.metropolis_step()
	BH_2.metropolis_step()
	tau.metropolis_step()

	if i%thin ==0:
		Rick_1.tally(i/thin)
		Rick_2.tally(i/thin)
		BH_1.tally(i/thin)
		BH_2.tally(i/thin)
		tau.tally(i/thin)


_plotter = PlotFactory()
Rick_1.plot(_plotter)
Rick_2.plot(_plotter)
BH_1.plot(_plotter)
BH_2.plot(_plotter)
tau.plot(_plotter)

print "Fraction of value computations skipped, Ricker curve: ",
print float(Ricker.value_computations) / float(Ricker.value_computations + Ricker.value_computation_skips)
print "Fraction of value computations skipped, Beverton-Holt curve ",
print float(BH.value_computations) / float(BH.value_computations + BH.value_computation_skips)

print "Fraction of prior computations skipped, Ricker data: ",
print float(data_Rick.prior_computations) / float(data_Rick.prior_computations + data_Rick.prior_computation_skips)
print "Fraction of prior computations skipped, Beverton-Holt data: ",
print float(data_BH.prior_computations) / float(data_BH.prior_computations + data_BH.prior_computation_skips)

c=raw_input()