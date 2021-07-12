#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# import aesara.tensor as at
import numpy as np
import pandas as pd

from pymc3.distributions import distribution
from pymc3.distributions.continuous import (
    Beta,
    Cauchy,
    ChiSquared,
    Exponential,
    Gamma,
    InverseGamma,
    Laplace,
    Normal,
    StudentT,
    Uniform,
)

# from aesara import scan
# from scipy import stats
#
# import pymc3 as pm


class SARIMA(distribution.Continuous):

    """
    .. math::
        SARIMA_{(p, d, q) \times (P, D, Q)_s} can be represented as

        \\phi_p (L) \tilde \\phi_P (L^s) \\Delta^d \\Delta_s^D y_t = A(t) +
        \theta_q (L) \tilde \theta_Q (L^s) \\zeta_t


    .. math::
    """

    def __init__(
        self, order, seasonal, ts=None, period=None, xreg=None, name=None, *args, **kwargs
    ):
        super().__init__()

        if ts is not None:
            self.y = np.array(ts).astype(float)
            self.yreal = pd.TimeSeries(ts)
            self.n = len(self.y)

        # series name
        if name is not None:
            sn = str(name)
        else:
            ts = dict()
            sn = f"{ts=}".split("=")[0]

        dimension = 1

        # order

        if order[1] >= 0:
            self.p = order[1]
        if order[2] >= 0:
            self.q = order[2]
        if order[3] >= 0:
            self.d = order[3]

        # Period
        if period == 0:
            self.period = self.yreal.freq
        else:
            self.period = period

        # Seasonal Order
        self.sp = seasonal[1]
        self.sq = seasonal[2]
        self.dd = seasonal[3]

        if self.period <= 1:
            self.pp = 0
            self.sq = 0
            self.dd = 0

        self.phi = np.array(self.p)  # ar parameters
        self.phi0 = np.linspace(self.p, -1, 1)

        self.sphi = np.array(self.sp)  # sar parameters
        self.sphi0 = np.linspace(self.sp, -1, 1)

        self.theta = np.array(self.q)  # ma parameters
        self.theta0 = np.linspace(self.q, -1, 1)

        self.stheta = np.array(self.sq)  # sma parameters
        self.stheta0 = np.linspace(self.sq, -1, 1)

        self.mu = np.array(self.n1)  # Mean pparameter
        self.epsilon = np.array(self.n1)  # residual parameter

        self.dinits = self.d + (self.period * self.dd)
        self.n1 = self.n

        if self.xreg is not None:
            assert (
                all(isinstance(ele, list) for ele in self.xreg) == True
            ), "xreg has to be a matrix with row dimension as same as the length of the time series"

            assert (
                self.xreg.shape[0] == self.n
            ), "The length of xreg don't match with the length of the time series"

            assert self.dd == 0, "seasonal difference is not allowed in dynamic regressions D  = 0 "
            self.dd = 0
            self.d1 = self.xreg.shape[1]
            self.breg = np.array(self.d1)

            if self.d > 0:
                self.xreg = np.diff(self.xreg, n=self.d)
                self.xlast = np.array((self.d, self.d1))
                self.xlast[
                    1,
                ] = self.xreg[-1]

            if self.d > 1:
                for i in range(2, self.d):
                    self.xlast[i,] = np.diff(
                        self.xreg, n=i - 1
                    )[-1]

        else:

            self.d1 = 0
            self.n2 = self.n1 - self.d - (self.period * self.dd)
            self.xreg = np.tile([0, self.d1 * self.n2], (self.d1, self.n2))

        # Default Priors
        self.prior_sigma0 = StudentT.dist(mu=0, sigma=1, nu=7)
        self.prior_mu0 = StudentT.dist(mu=0, sigma=2.5, nu=6)
        self.prior_ar = Normal.dist(mu=0, sigma=0.5)
        self.prior_ma = Normal.dist(mu=0, sigma=0.5)
        self.prior_sar = Normal.dist(mu=0, sigma=0.5)
        self.prior_sma = Normal.dist(mu=0, sigma=0.5)
        self.prior_breg = StudentT.dist(mu=0, sigma=2.5, nu=6)

    def transformation_coefficients(self, number, hyperparameter, parameter, parameter0):
        for i in range(number):
            if hyperparameter[i, 4] == 1:
                parameter[i] = parameter0[i]
            else:
                parameter[i] = 2 * abs(parameter0[i]) - 1

        return

    def transform(self):
        self.phi = self.transformation(self.p, self.phi, self.phi0, self.prior_ar)
        self.theta = self.transformation(self.q, self.theta, self.theta0, self.prior_ma)
        self.sphi = self.transformation(self.sp, self.sphi, self.sphi0, self.prior_sar)
        self.stheta = self.transformation(self.sq, self.stheta, self.stheta0, self.prior_sma)

    def arma_estimation(self):
        if self.d1 > 0:
            mu = self.xreg * self.breg
        else:
            mu = np.zeros(self.n)

        for i in range(self.n):
            mu[i] += self.mu0

            for j in range(self.p):
                if i > j:
                    mu[i] += self.y[i - j] * self.phi[j]

            for j in range(self.q):
                if i > j:
                    mu[i] += self.epsilon[i - j] * self.theta[j]

            for j in range(self.sp):
                if i > (self.period * j):
                    mu[i] += self.y[i - (self.period * j)] * self.sphi[j]

            for j in range(self.sq):
                if i > (self.period * j):
                    mu[i] += self.epsilon[i - (self.period * j)] * self.stheta[j]

    def invchi(self, x, chisquare):
        return (1.0 / x ** 2) * chisquare.logp(1 / x)

    def specify_priors(self):
        def priorint(self, prior, x, target, n, dist):
            if prior == n:
                target += dist.logp(x)

        def target_hyperparam(self, target, prior_x0, prior_x01, prior_x02, prior_x03, x0):
            # prior mu
            self.priorint(prior_x0, x0, target, 1, Normal.dist(mu=prior_x01, sigma=prior_x02))
            self.priorint(prior_x0, x0, target, 2, Beta.dist(alpha=prior_x01, beta=prior_x02))
            self.priorint(prior_x0, x0, target, 3, Uniform.dist(lower=prior_x01, upper=prior_x02))
            self.priorint(
                prior_x0,
                x0,
                target,
                4,
                StudentT.dist(mu=prior_x01, sigma=prior_x02, nu=prior_x03),
            )
            self.priorint(prior_x0, x0, target, 5, Cauchy.dist(alpha=prior_x01, beta=prior_x02))
            self.priorint(
                prior_x0, x0, target, 7, InverseGamma.dist(alpha=prior_x01, beta=prior_x02)
            )
            if self.prior_x0 == 7:
                target += self.invchi(x0, ChiSquared.dist(nu=prior_x03))  # inverse chi square
            if self.prior_x0 == 8:
                target += -np.log(self.sigma0)
            self.priorint(prior_x0, x0, target, 9, Gamma.dist(alpha=prior_x01, beta=prior_x03))
            self.priorint(prior_x0, x0, target, 10, Exponential.dist(lam=prior_x02))
            self.priorint(prior_x0, x0, target, 11, ChiSquared.dist(nu=prior_x03))
            self.priorint(prior_x0, x0, target, 12, Laplace.dist(mu=prior_x01, b=prior_x02))

        def prior_for_sigma0(self, target, prior_sigma0, sigma0):
            self.target_hyperparam(
                target, prior_sigma0[4], prior_sigma0[1], prior_sigma0[2], prior_sigma0[3], sigma0
            )
            return target

        def prior_for_mu0(self, target, prior_mu0, mu0):
            self.target_hyperparam(
                target, prior_mu0[4], prior_mu0[1], prior_mu0[2], prior_mu0[3], mu0
            )
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
            self.priorint(prior_x0, x0, target, 1, Normal.dist(mu=prior_x01, sigma=prior_x02))
            self.priorint(prior_x0, x0, target, 2, Beta.dist(alpha=prior_x01, beta=prior_x02))
            self.priorint(prior_x0, x0, target, 3, Uniform.dist(lower=prior_x01, upper=prior_x02))

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

        def prior_sar(self, sp, target, prior_sar, sphi0):
            return self.ar_ma_priors(sp, target, prior_sar, sphi0)

        def prior_sma(self, sq, target, prior_sma, stheta0):
            return self.ar_ma_priors(sq, target, prior_sma, stheta0)

    def get_order_arima(self):
        return dict(
            {
                "p": self.p,
                "d": self.d,
                "q": self.q,
                "sp": self.sp,
                "dd": self.dd,
                "sq": self.sq,
                "d1": self.d1,
                "period": self.period,
            }
        )

    def max_order_arima(self):
        return max(self.p, self.q, self.period, self.sp, self.period * self.sq)

    def likelihood(self, target):
        target += Normal(mu=0, sigma=self.sigma0).logp(self.epsilon)

    def generated_quantities(self):
        loglik = 0
        log_lik = np.array(self.n1)
        fit = np.array(self.n)
        residuals = np.array(self.n)

        for i in range(self.n):
            if i <= self.dinits:
                residuals[i] = Normal(mu=0, sigma=self.sigma0)
            else:
                residuals[i] = Normal(mu=self.epsilon[i - self.dinits], sigma=self.sigma0)
            fit[i] = self.yreal[i] - residuals[i]
            if i <= self.n1:
                log_lik[i] = Normal(mu=self.mu[i], sigma=self.sigma0).logp(self.y[i])
                loglik += log_lik[i]
