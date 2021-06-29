import numpy as np
import pymc3 as pm

class Sarima:

    def __init__(
            self, n, n1, p, d, q, pp, dd, qq, period, d1, sigma0, mu0
   ):
        # Model data
        self.n = n          # number of data items
        self.n1 = n1         # number of data items
        self.p = p          # number of predictors  ar
        self.d = d          # number of simple differences
        self.q = q         # number of predictions ma
        self.pp = pp      # number of predictors  arch
        self.dd = dd        # number of seasonal differences
        self.qq = qq        # number of predictions GARCH
        self.period = period    # time series period
        self.d1 = d1         # number of independent variables
        self.sigma0 = sigma0   # Variance parameter
        self.mu0 = mu0              # location parameter

        # data

        self.y = np.array(self.n1)          # outcome time series
        self.yreal = np.array(self.n)     # outcome not differenced time series
        self.xreg = np.array((self.n1,self.d1))      # matrix with independent variables

        #prior data

        self.prior_mu0 = np.array(4)     # prior location parameter
        self.prior_sigma0 = np.array(4)  # prior scale parameter
        self.prior_ar = np.array((self.p,4))  # ar location hyper parameters
        self.prior_ma = np.array((self.q,4))    # ma location hyper parameters
        self.prior_sar = np.array((self.pp, 4)) # prior arch hyper parameters
        self.prior_sma = np.array((self.qq, 4))   # prior ma hyper parameters
        self.prior_breg = np.array((self.d1, 4)) # prior ma hyper parameters
        self.dinits = self.d+(self.period*self.dd)

    # transformed_data

    # parameters
        self.breg = np.array(self.d1)      # regression parameters
        self.phi0 = np.linspace(self.p, -1, 1) # ar parameters
        self.theta0 = np.linspace(self.q, -1, 1) # ma parameters
        self.pphi0 = np.linspace(self.pp, -1, 1)  # sar parameters
        self.Theta0 = np.linspace(self.qq, -1, 1) # sma parameters

    # model_parameters
        self.phi = np.array(self.p)   # ar parameters
        self.theta = np.array(self.q)  # ma parameters
        self.sphi = np.array(self.pp)    # ar parameters
        self.stheta = np.array(self.qq)  # ma parameters
        self.mu = np.array(self.n1)       # Mean pparameter
        self.epsilon = np.array(self.n1)     # residual parameter

    def transformation_coefficients(self):

        for i in range(self.p):
            if self.prior_ar[i, 4] ==1:
                self.phi[i] = self.phi0[i]
            else:
                self.phi[i] = (2 * abs(self.phi0[i]) - 1)

        for i in range(self.q):
            if self.prior_ma[i, 4] ==1:
                self.theta[i] = self.theta0[i]
            else:
                 self.theta[i] = (2 * abs(self.theta0[i]) - 1)

        for i in range(self.pp):
            if self.prior_sar[i, 4] == 1:
                self.sphi[i] = self.pphi0[i]
            else:
                self.sphi[i] = (2 * abs(self.pphi0[i]) - 1)

        for i in range(self.qq):
            if self.prior_sma[i, 4] == 1:
                self.stheta[i] = self.Theta0[i]
            else:
                self.stheta[i] = (2 * abs(self.Theta0[i]) - 1)

    # model

    def prior_for_mu0(self, target,prior_for_mu0):
        #prior mu
        if self.prior_mu0[4] == 1:
            target += pm.Normal(mu=self.prior_mu0[1], sigma=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 2:
            target += pm.Beta(alpha=self.prior_mu0[1], beta=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 3:
            target += pm.Uniform(lower=self.prior_mu0[1], upper=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 4:
            target += pm.StudentT(nu=self.prior_mu0[3], mu=self.prior_mu0[1], sigma=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 5:
            target += pm.Cauchy(alpha=self.prior_mu0[1], beta=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 6:
            target += pm.InverseGamma(alpha=self.prior_mu0[1], beta=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 7:
            target += pm.ChiSquared(nu=self.prior_mu0[3]).logp(self.mu0) #inverse actually
        if self.prior_mu0[4] == 8:
            target += -np.log(self.sigma0)
        if self.prior_mu0[4] == 9:
            target += pm.Gamma(alpha=self.prior_mu0[1], beta=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 10:
            target += pm.Exponential(lam=self.prior_mu0[2]).logp(self.mu0)
        if self.prior_mu0[4] == 11:
            target += pm.ChiSquared(nu=self.prior_mu0[3]).logp(self.mu0)
        if self.prior_mu0[4] == 12:
            target += pm.Laplace(mu=self.prior_mu0[1], b=self.prior_mu0[2]).logp(self.mu0)

        #prior sigma

        if self.prior_sigma0[4] == 1:
            target += pm.Normal(mu=self.prior_sigma0[1], sigma=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 2:
            target += pm.Beta(alpha=self.prior_sigma0[1], beta=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 3:
            target += pm.Uniform(lower=self.prior_sigma0[1], upper=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 4:
            target += pm.StudentT(nu=self.prior_sigma0[3], mu=self.prior_sigma0[1], sigma=self.prior_sigma0[2]).logp(
                self.sigma0)
        if self.prior_sigma0[4] == 5:
            target += pm.Cauchy(alpha=self.prior_sigma0[1], beta=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 6:
            target += pm.InverseGamma(alpha=self.prior_sigma0[1], beta=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 7:
            target += pm.ChiSquared(nu=self.prior_sigma0[3]).logp(self.sigma0) #inverse actually
        if self.prior_sigma0[4] == 8:
            target += -np.log(self.sigma0)
        if self.prior_sigma0[4] == 9:
            target += pm.Gamma(alpha=self.prior_sigma0[1], beta=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 10:
            target += pm.Exponential(lam=self.prior_sigma0[2]).logp(self.sigma0)
        if self.prior_sigma0[4] == 11:
            target += pm.ChiSquared(nu=self.prior_sigma0[3]).logp(self.sigma0)
        if self.prior_sigma0[4] == 12:
            target += pm.Laplace(mu=self.prior_sigma0[1], b=self.prior_sigma0[2]).logp(self.sigma0)

        # prior breg
        if self.d1>0:
            for i in range(self.d1):
                if self.prior_breg[i, 4] == 1:
                    target += pm.Normal(mu=self.prior_breg[i, 1], sigma=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 2:
                    target += pm.Beta(alpha=self.prior_breg[i, 1], beta=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 3:
                    target += pm.Uniform(lower=self.prior_breg[i, 1], upper=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 4:
                    target += pm.StudentT(nu=self.prior_breg[i, 3], mu=self.prior_breg[i, 1], sigma=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 5:
                    target += pm.Cauchy(alpha=self.prior_breg[i, 1], beta=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 6:
                    target += pm.InverseGamma(alpha=self.prior_breg[i, 1], beta=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 7:
                    target += pm.ChiSquared(nu=self.prior_breg[i, 3]).logp(self.breg[i]) #inverse actually
                if self.prior_breg[i, 4] == 8:
                    target += np.log(self.sigma0)
                if self.prior_breg[i, 4] == 9:
                    target += pm.Gamma(alpha=self.prior_breg[i, 1], beta=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 10:
                    target += pm.Normal(lam=self.prior_breg[i, 2]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 11:
                    target += pm.ChiSquared(nu=self.prior_breg[i, 3]).logp(self.breg[i])
                if self.prior_breg[i, 4] == 12:
                    target += pm.Laplace(mu= self.prior_breg[i, 1], b=self.prior_breg[i, 2]).logp(self.breg[i])
                
                # prior ar
                if self.p>0:
                    for i in range(self.p):
                        if self.prior_ar[i,4]==1:
                            target += pm.Normal(mu=self.prior_ar[i,1],sigma=self.prior_ar[i,2]).logp(self.phi0[i])
                        if self.prior_ar[i,4]==2:
                            target += pm.Beta(alpha=self.prior_ar[i, 1], beta=self.prior_ar[i, 2]).logp(self.phi0[i])
                        if self.prior_ar[i,4]==3:
                            target += pm.Uniform(lower=self.prior_ar[i, 1], upper=self.prior_ar[i, 2]).logp(self.phi0[i])
                # prior ma
                if self.q>0:
                    for i in range(self.q):
                        if self.prior_ma[i,4]==1:
                            target += pm.Normal(mu=self.prior_ma[i,1],sigma=self.prior_ma[i,2]).logp(self.theta0[i])
                        if self.prior_ma[i,4]==2:
                            target += pm.Beta(alpha=self.prior_ma[i, 1], beta=self.prior_ma[i, 2]).logp(self.theta0[i])
                        if self.prior_ma[i,4]==3:
                            target += pm.Uniform(lower=self.prior_ma[i, 1], upper=self.prior_ma[i, 2]).logp(self.theta0[i])

                # prior sar
                if self.pp>0:
                    for i in range(self.pp):
                        if self.prior_sar[i,4]==1:
                            target += pm.Normal(mu=self.prior_sar[i,1],sigma=self.prior_sar[i,2]).logp(self.pphi0[i])
                        if self.prior_sar[i,4]==2:
                            target += pm.Beta(alpha=self.prior_sar[i, 1], beta=self.prior_sar[i, 2]).logp(self.pphi0[i])
                        if self.prior_sar[i,4]==3:
                            target += pm.Uniform(lower=self.prior_sar[i, 1], upper=self.prior_sar[i, 2]).logp(self.pphi0[i])
                # prior sma
                if self.qq>0:
                    for i in range(self.qq):
                        if self.prior_sma[i,4]==1:
                            target += pm.Normal(mu=self.prior_sma[i,1],sigma=self.prior_sma[i,2]).logp(self.ttheta0[i])
                        if self.prior_sma[i,4]==2:
                            target += pm.Beta(alpha=self.prior_sma[i, 1], beta=self.prior_sma[i, 2]).logp(abs(self.ttheta0[i]))
                        if self.prior_sma[i,4]==3:
                            target += pm.Uniform(lower=self.prior_sma[i, 1], upper=self.prior_sma[i, 2]).logp(self.ttheta0[i])

                # likelihood
                target += pm.Normal(mu=0, sigma=self.sigma0).logp(self.epsilon)

                # generated quantities
                loglik=0
                log_lik = np.array(self.n1)
                fit = np.array(self.n)
                residuals = np.array(self.n)

                for i in range(self.n):
                    if i<=self.dinits:
                        residuals[i] = pm.Normal(mu=0, sigma=self.sigma0)
                    else:
                        residuals[i] = pm.Normal(mu=self.epsilon[i-self.dinits], sigma=self.sigma0)
                    fit[i] = self.yreal[i] - residuals[i]
                    if i <= self.n1:
                        log_lik[i] = pm.Normal(mu=self.mu[i], sigma=self.sigma0).logp(self.y[i])
                        loglik += log_lik[i]

                