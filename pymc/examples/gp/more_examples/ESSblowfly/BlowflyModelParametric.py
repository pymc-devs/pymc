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
from TS_fortran import ts_fortran

T=len(blowflydata)*2 + 100
t = arange(0.,T)
obs_slice = arange(0,len(blowflydata)*2,2,dtype=int)

dxsteer = 800.
xsteer = arange(0., max(blowflydata)+dxsteer, dxsteer)
xlpot = xplot=arange(0.,max(blowflydata),100.)


#######
#  B  #
#######

@stoch
def RickerRate(value= 0.00268, alpha=3., mean=.002):
    """The mean for the birth rate is a Ricker function"""
    return gamma_like(value, alpha, mean/alpha)

@stoch
def RickerSlope(value=11.93, alpha=3., mean=4.):
    """RickerSlope ~ distribution(alpha, mean)"""
    return gamma_like(value, alpha, mean/alpha)

#######
#  D  #
#######
# 
@stoch
def MortalityMean(value=.1551, alpha=3., mean=.1):
    """The mean for the per-capita mortality rate is centered on 1"""
    return gamma_like(value, alpha, mean/alpha)


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


@stoch
def mu_psi(value=-.640):
    """mu_psi ~ normal(mu, V)"""
    return 0.

@stoch
def V_psi(value=.537):
    """V_psi ~ gamma(alpha, mean)"""
    return 0.

@stoch
def psi(value=exp(normal(size=T)*sqrt(V_psi) + mu_psi), mu=mu_psi, V=V_psi):
    """psi ~ lognormal(mu, V)"""
    return lognormal_like(value, mu, 1./V)


#######
# phi #
#######

@stoch
def mu_phi(value=-.532):
    """mu_phi ~ normal(mu, V)"""
    return 0.

@stoch
def V_phi(value=.695):
    """V_phi ~ gamma(alpha, mean)"""
    return 0.

@stoch
def phi(value=exp(normal(size=T)*sqrt(V_phi) + mu_phi), mu=mu_phi, V=V_phi):
    """phi ~ lognormal(mu, V)"""
    return lognormal_like(value, mu, 1./V)


##########################
# Unobserved time series #
##########################
@stoch
def TS_IC(value=blowflydata[:tau.value], alpha=3., mean= 500.):
    """TS_IC ~ distribution(parents)"""
    return gamma_like(value, alpha, mean/alpha)

@dtrm
def TS(M = MortalityMean, S = RickerSlope, r = RickerRate, psi=psi, phi=phi, tau=tau, IC = TS_IC):
    """TS = function(B, D, psi, phi, t)"""
    return ts_fortran(M,r,S,IC,psi,phi)
    
########
# Data #
########

@stoch
def meas_V(value=.0798, mean=.01, alpha=3.):
    """meas_V ~ gamma(mean, alpha)"""
    return gamma_like(value, alpha, mean/alpha)

@data
@stoch
def TSdata(value=blowflydata, mu=TS, V=meas_V):
    """TSdata ~ normal(mu, V)"""
    return normal_like(log(value), log(mu[obs_slice]), 1./V)

J = JointMetropolis(  [phi, psi, MortalityMean, RickerRate, RickerSlope, TS_IC, mu_phi, mu_psi, V_phi, V_psi],
                        epoch=1,
                        memory=1,
                        delay=0,
                        scale=.1,
                        oneatatime_scales = {phi: .01,
                                            psi: .01,
                                            MortalityMean: .1,
                                            RickerRate: 1.,
                                            RickerSlope: 1.,
                                            TS_IC: .05,
                                            mu_phi: 1.,
                                            mu_psi: 1.,
                                            V_phi: 1.,
                                            V_psi: 1.})
                                        
# phiStepper = Metropolis(phi, scale=.01)
# psiStepper = Metropolis(psi, scale=.01)  
# MortMeanStepper = Metropolis(MortalityMean, scale=.1)
# RickRateStepper = Metropolis(RickerRate, scale=1.)
# RickSlopeStepper = Metropolis(RickerSlope, scale=1.)
# TSICStepper = Metropolis(TS_IC, scale=.05)
# muphiStepper = Metropolis(mu_phi, scale=1.)
# mupsiStepper = Metropolis(mu_psi, scale=1.)
# VphiStepper = Metropolis(V_phi, scale=1.)
# VpsiStepper = Metropolis(V_psi, scale=1.)
# measVStepper = Metropolis(meas_V, scale=1.)