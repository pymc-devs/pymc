# Also do a run with zero mean, show that the amp stoch doesn't go to zero.
# interpretation of amp stoch going to zero is that the stochetric means
# do just as good of a job of explaining the data as the nonstochetric functions
# in this case. You needed the flexible model to see that; in the zero-mean case
# that won't happen of course.

from pymc.gp import *
from pymc.gp.cov_funs import gaussian, fourier_basis
from pymc import *
from pylab import *
from numpy import *
from numpy.random import normal, poisson
from scipy import special
from blowflydata import blowflydata

T=len(blowflydata) + 100
t = arange(0.,T)
obs_slice = arange(0,len(blowflydata),dtype=int)

dxsteer = 800.
xsteer = arange(0., max(blowflydata)+dxsteer, dxsteer)
xlpot = xplot=arange(0.,max(blowflydata),100.)


#######
#  B  #
#######
    
RickerRate = Gamma('RickerRate', .00225, alpha=3., beta=.002/3., isdata=True)
RickerSlope = Gamma('RickerSlope', 5., alpha=3., beta=4./3., isdata=True)

# # Uncomment for nonparametric B
B_amp = Gamma('B_amp', .4, alpha=5., beta=1./4., isdata=True)
B_scale = 10000.

@dtrm
def B_C(amp=B_amp, scale=B_scale):
    """B_C = function(amp, scale)"""
    C = BasisCovariance(basis = fourier_basis, scale = scale, coef_cov = diag(amp**2/arange(1,15,dtype=float)**((2.5))))
    return C

@dtrm
def B_M(rate = RickerRate, slope=RickerSlope):
    """The mean function decays"""
    M = Mean(lambda x: log(slope) - rate * x)
    return M

B = GP(B_M, B_C, mesh = xsteer, name = 'B', init_mesh_vals = B_M.value(xsteer), trace=True)


# Uncomment for parametric B
# @dtrm
# def B(rate = RickerRate, slope=RickerSlope):
#     """The mean function decays"""
#     return lambda x: log(slope) - rate * x


#######
#  D  #
#######

MortalityMean = Gamma('MortalityMean', .14, alpha=3., beta=.1/alpha, isdata=True)

# # Uncomment for nonparametric D
D_amp = Gamma('D_amp', .12, alpha=4., beta=.1/5., isdata=True)
D_scale = 1000.

@dtrm
def D_C(amp=D_amp, scale=D_scale):
    """D_C = function(amp, scale)"""
    C = BasisCovariance(basis = fourier_basis, scale = scale, coef_cov = diag(amp**2/arange(1,25,dtype=float)**((1.5))))
    return C

@dtrm
def D_M(mean=MortalityMean):
    """The mean function is constant at 1"""
    M = Mean(lambda x: 0.*x+log(mean))
    return M

D = GP(D_M, D_C, mesh = xsteer, name = 'D', init_mesh_vals = D_M.value(xsteer))


# Uncomment for parametric D
# @dtrm
# def D(mean=MortalityMean):
#     """The mean function is constant at 1"""
#     return lambda x: 0.*x+log(mean)


#######
# tau #
#######

@data
@discrete_stoch
def tau(value=14, min=2, max=50):
    """The delay"""
    if value<min or value>max:
        return -Inf
    else:
        return 0.



#######
# psi #
#######

psi_init = ones(T,dtype=float)
for s in range(1, len(blowflydata)):
    psi_init[s-1] = blowflydata[s] / (blowflydata[s-1] + \
    exp(B.value(blowflydata[s-tau.value])) * blowflydata[s-tau.value] -\
    exp(D.value(blowflydata[s-1])) * blowflydata[s-1])

mu_psi = Uniform('mu_psi', -.1, lower=-Inf, upper=Inf)
V_psi = Uniform('V_psi', .4, lower=-Inf, upper=Inf)
psi = Lognormal('psi', psi_init, mu=mu_psi, V=V_psi)    


##########################
# Unobserved time series #
##########################

TS_IC = Gamma('TS_IC', blowflydata[:tau.value], alpha/3., beta=500./3.)

@dtrm
def TS(B=B, D=D, psi=psi, t=t, tau=tau, IC = TS_IC):
    """TS = function(B, D, psi, phi, t)"""
    value = zeros(T,dtype=float)
    value[:tau] = IC

    for s in xrange(tau,T):

        value[s] = (value[s-1] + exp(B(value[s-tau]))* value[s-tau] - exp(D(value[s-1])) * value[s-1]) * psi[s-1]
    
        if value[s]<0. or isnan(value[s]):
            value[s]=0.
            break
    return value


########
# Data #
########

meas_V = Gamma('meas_V', .01, alpha=3., beta=.01/3., isdata=True)
TSdata = Normal('TSdata', blowflydata, mu=TS, V=meas_V, isdata=True)

PsiStepper = JointMetropolis([psi, mu_psi, V_psi, TS_IC], 
            oneatatime_scales = {psi: .005, mu_psi: 1., V_psi: 1., TS_IC: .01},
            epoch = 50)
            
# RickRateSlopeStepper = GPParentMetropolis(metro_method = JointMetropolis([RickerRate, RickerSlope], oneatatime_scales = {RickerRate: .1, RickerSlope: .1}))
BSampler = GPMetropolis(B,scale=.01,verbose=False)
DSampler = GPMetropolis(D,scale=.01,verbose=False)
# MortMeanStepper = GPParentMetropolis(metro_method=Metropolis(MortalityMean, scale=.1))
# RickRateStepper = GPParentMetropolis(metro_method=Metropolis(RickerRate, scale=.1))
# RickSlopeStepper = GPParentMetropolis(metro_method=Metropolis(RickerSlope, scale=.1))
