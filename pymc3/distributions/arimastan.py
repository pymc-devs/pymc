import numpy as np

import pymc3 as pm


def invchi(x, chisquare):
    return (1.0 / x ** 2) * chisquare.logp(1 / x)


class SARIMA:
    def __init__(
        self,
        n,
        n1,
        p,
        d,
        q,
        pp,
        dd,
        qq,
        period,
        d1,
        sigma0,
        mu0,
    ):
        # Model data
        self.n = n  # number of data items
        self.n1 = n1  # number of data items
        self.p = p  # number of predictors  ar
        self.d = d  # number of simple differences
        self.q = q  # number of predictions ma
        self.pp = pp  # number of predictors  arch
        self.dd = dd  # number of seasonal differences
        self.qq = qq  # number of predictions GARCH
        self.period = period  # time series period
        self.d1 = d1  # number of independent variables
        self.sigma0 = sigma0  # Variance parameter
        self.mu0 = mu0  # location parameter

        # data

        self.y = np.array(self.n1)  # outcome time series
        self.yreal = np.array(self.n)  # outcome not differenced time series
        self.xreg = np.array((self.n1, self.d1))  # matrix with independent variables

        # prior data

        self.prior_mu0 = np.array(4)  # prior location parameter
        self.prior_sigma0 = np.array(4)  # prior scale parameter
        self.prior_ar = np.array((self.p, 4))  # ar location hyper parameters
        self.prior_ma = np.array((self.q, 4))  # ma location hyper parameters
        self.prior_sar = np.array((self.pp, 4))  # prior arch hyper parameters
        self.prior_sma = np.array((self.qq, 4))  # prior ma hyper parameters
        self.prior_breg = np.array((self.d1, 4))  # prior ma hyper parameters
        self.dinits = self.d + (self.period * self.dd)

        # transformed_data

        # parameters
        self.breg = np.array(self.d1)  # regression parameters
        self.phi0 = np.linspace(self.p, -1, 1)  # ar parameters
        self.theta0 = np.linspace(self.q, -1, 1)  # ma parameters
        self.pphi0 = np.linspace(self.pp, -1, 1)  # sar parameters
        self.ttheta0 = np.linspace(self.qq, -1, 1)  # sma parameters

        # model_parameters
        self.phi = np.array(self.p)  # ar parameters
        self.theta = np.array(self.q)  # ma parameters
        self.sphi = np.array(self.pp)  # ar parameters
        self.stheta = np.array(self.qq)  # ma parameters
        self.mu = np.array(self.n1)  # Mean pparameter
        self.epsilon = np.array(self.n1)  # residual parameter

    def transformation(self, number, parameter, parameter0, hyperparameter):

        for i in range(number):
            if hyperparameter[i, 4] == 1:
                parameter[i] = parameter0[i]
            else:
                parameter[i] = 2 * abs(parameter0[i]) - 1

    def transformation_coefficients(
        self,
        p,
        phi,
        phi0,
        prior_ar,
        q,
        theta,
        theta0,
        prior_ma,
        pp,
        sphi,
        pphi0,
        prior_sar,
        qq,
        stheta,
        ttheta0,
        prior_sma,
    ):

        self.transformation(p, phi, phi0, prior_ar)
        self.transformation(q, theta, theta0, prior_ma)
        self.transformation(pp, sphi, pphi0, prior_sar)
        self.transformation(qq, stheta, ttheta0, prior_sma)

    def priorint(self, prior, x, target, n, dist):
        if prior == n:
            target += dist.logp(x)

    def target_hyperparam(self, target, prior_x0, prior_x01, prior_x02, prior_x03, x0):
        # prior mu
        self.priorint(prior_x0, x0, target, 1, pm.Normal.dist(mu=prior_x01, sigma=prior_x02))
        self.priorint(prior_x0, x0, target, 2, pm.Beta.dist(alpha=prior_x01, beta=prior_x02))
        self.priorint(prior_x0, x0, target, 3, pm.Uniform.dist(lower=prior_x01, upper=prior_x02))
        self.priorint(
            prior_x0,
            x0,
            target,
            4,
            pm.StudentT.dist(mu=prior_x01, sigma=prior_x02, nu=prior_x03),
        )
        self.priorint(prior_x0, x0, target, 5, pm.Cauchy.dist(alpha=prior_x01, beta=prior_x02))
        self.priorint(
            prior_x0, x0, target, 7, pm.InverseGamma.dist(alpha=prior_x01, beta=prior_x02)
        )
        if self.prior_x0 == 7:
            target += invchi(x0, pm.ChiSquared.dist(nu=prior_x03))  # inverse chi square
        if self.prior_x0 == 8:
            target += -np.log(self.sigma0)
        self.priorint(prior_x0, x0, target, 9, pm.Gamma.dist(alpha=prior_x01, beta=prior_x03))
        self.priorint(prior_x0, x0, target, 10, pm.Exponential.dist(lam=prior_x02))
        self.priorint(prior_x0, x0, target, 11, pm.ChiSquared.dist(nu=prior_x03))
        self.priorint(prior_x0, x0, target, 12, pm.Laplace.dist(mu=prior_x01, b=prior_x02))

    def prior_for_sigma0(self, target, prior_sigma0, sigma0):
        self.target_hyperparam(
            target, prior_sigma0[4], prior_sigma0[1], prior_sigma0[2], prior_sigma0[3], sigma0
        )
        return target

    def prior_for_mu0(self, target, prior_mu0, mu0):
        self.target_hyperparam(target, prior_mu0[4], prior_mu0[1], prior_mu0[2], prior_mu0[3], mu0)
        return target

    def prior_for_breg(self, target, prior_breg, breg):
        if self.d1 > 0:
            for i in range(self.d1):
                self.target_hyperparam(
                    target,
                    prior_breg[i, 4],
                    prior_breg[i, 1],
                    prior_breg[i, 2],
                    prior_breg[i, 3],
                    breg[i],
                )
        return target

    def target_hyperparam3(self, target, prior_x0, prior_x01, prior_x02, x0):
        # prior mu
        self.priorint(prior_x0, x0, target, 1, pm.Normal.dist(mu=prior_x01, sigma=prior_x02))
        self.priorint(prior_x0, x0, target, 2, pm.Beta.dist(alpha=prior_x01, beta=prior_x02))
        self.priorint(prior_x0, x0, target, 3, pm.Uniform.dist(lower=prior_x01, upper=prior_x02))

    def ar_ma_priors(self, x, target, prior_ar_ma, ar_ma):
        if x > 0:
            for i in range(x):
                self.target_hyperparam3(
                    target, prior_ar_ma[i, 4], prior_ar_ma[i, 1], prior_ar_ma[i, 2], ar_ma[i]
                )

    def prior_ar(self, p, target, prior_ar, phi0):
        return self.ar_ma_priors(p, target, prior_ar, phi0)

    def prior_ma(self, q, target, prior_ma, theta0):
        return self.ar_ma_priors(q, target, prior_ma, theta0)

    def prior_sar(self, pp, target, prior_sar, pphi0):
        return self.ar_ma_priors(pp, target, prior_sar, pphi0)

    def prior_sma(self, qq, target, prior_sma, ttheta0):
        return self.ar_ma_priors(qq, target, prior_sma, ttheta0)

    def likelihood(self, target):
        target += pm.Normal(mu=0, sigma=self.sigma0).logp(self.epsilon)

    def generated_quantities(self):
        loglik = 0
        log_lik = np.array(self.n1)
        fit = np.array(self.n)
        residuals = np.array(self.n)

        for i in range(self.n):
            if i <= self.dinits:
                residuals[i] = pm.Normal(mu=0, sigma=self.sigma0)
            else:
                residuals[i] = pm.Normal(mu=self.epsilon[i - self.dinits], sigma=self.sigma0)
            fit[i] = self.yreal[i] - residuals[i]
            if i <= self.n1:
                log_lik[i] = pm.Normal(mu=self.mu[i], sigma=self.sigma0).logp(self.y[i])
                loglik += log_lik[i]
