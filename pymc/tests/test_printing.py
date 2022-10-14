import numpy as np

from pymc import Bernoulli, Censored, Mixture
from pymc.aesaraf import floatX
from pymc.distributions import (
    DirichletMultinomial,
    HalfNormal,
    KroneckerNormal,
    MvNormal,
    NegativeBinomial,
    Normal,
    Uniform,
    ZeroInflatedPoisson,
)
from pymc.math import dot
from pymc.model import Deterministic, Model, Potential


# TODO: This test is a bit too monolithic
class TestStrAndLatexRepr:
    def setup_class(self):
        # True parameter values
        alpha, sigma = 1, 1
        beta = [1, 2.5]

        # Size of dataset
        size = 100

        # Predictor variable
        X = np.random.normal(size=(size, 2)).dot(np.array([[1, 0], [0, 0.2]]))

        # Simulate outcome variable
        Y = alpha + X.dot(beta) + np.random.randn(size) * sigma
        with Model() as self.model:
            # TODO: some variables commented out here as they're not working properly
            # in v4 yet (9-jul-2021), so doesn't make sense to test str/latex for them

            # Priors for unknown model parameters
            alpha = Normal("alpha", mu=0, sigma=10)
            b = Normal("beta", mu=0, sigma=10, size=(2,), observed=beta)
            sigma = HalfNormal("sigma", sigma=1)

            # Test Cholesky parameterization
            Z = MvNormal("Z", mu=np.zeros(2), chol=np.eye(2), size=(2,))

            # NegativeBinomial representations to test issue 4186
            # nb1 = pm.NegativeBinomial(
            #     "nb_with_mu_alpha", mu=pm.Normal("nbmu"), alpha=pm.Gamma("nbalpha", mu=6, sigma=1)
            # )
            nb2 = NegativeBinomial("nb_with_p_n", p=Uniform("nbp"), n=10)

            # SymbolicRV
            zip = ZeroInflatedPoisson("zip", 0.5, 5)

            # Nested SymbolicRV
            comp_1 = ZeroInflatedPoisson.dist(0.5, 5)
            comp_2 = Censored.dist(Bernoulli.dist(0.5), -1, 1)
            nested_mix = Mixture("nested_mix", [0.5, 0.5], [comp_1, comp_2])

            # Expected value of outcome
            mu = Deterministic("mu", floatX(alpha + dot(X, b)))

            # add a bounded variable as well
            # bound_var = Bound(Normal, lower=1.0)("bound_var", mu=0, sigma=10)

            # KroneckerNormal
            n, m = 3, 4
            covs = [np.eye(n), np.eye(m)]
            kron_normal = KroneckerNormal("kron_normal", mu=np.zeros(n * m), covs=covs, size=n * m)

            # MatrixNormal
            # matrix_normal = MatrixNormal(
            #     "mat_normal",
            #     mu=np.random.normal(size=n),
            #     rowcov=np.eye(n),
            #     colchol=np.linalg.cholesky(np.eye(n)),
            #     size=(n, n),
            # )

            # DirichletMultinomial
            dm = DirichletMultinomial("dm", n=5, a=[1, 1, 1], size=(2, 3))

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

            # add a potential as well
            pot = Potential("pot", mu**2)

        self.distributions = [alpha, sigma, mu, b, Z, nb2, zip, nested_mix, Y_obs, pot]
        self.deterministics_or_potentials = [mu, pot]
        # tuples of (formatting, include_params
        self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            ("plain", True): [
                r"alpha ~ Normal(0, 10)",
                r"sigma ~ HalfNormal(0, 1)",
                r"mu ~ Deterministic(f(beta, alpha))",
                r"beta ~ Normal(0, 10)",
                r"Z ~ MvNormal(f(), f())",
                r"nb_with_p_n ~ NegBinom(10, nbp)",
                r"zip ~ MarginalMixture(f(), DiracDelta(0), Poisson(5))",
                (
                    r"nested_mix ~ MarginalMixture(<constant>, "
                    r"MarginalMixture(f(), DiracDelta(0), Poisson(5)), "
                    r"Censored(Bern(0.5), -1, 1))"
                ),
                r"Y_obs ~ Normal(mu, sigma)",
                r"pot ~ Potential(f(beta, alpha))",
            ],
            ("plain", False): [
                r"alpha ~ Normal",
                r"sigma ~ HalfNormal",
                r"mu ~ Deterministic",
                r"beta ~ Normal",
                r"Z ~ MvNormal",
                r"nb_with_p_n ~ NegBinom",
                r"zip ~ MarginalMixture",
                r"nested_mix ~ MarginalMixture",
                r"Y_obs ~ Normal",
                r"pot ~ Potential",
            ],
            ("latex", True): [
                r"$\text{alpha} \sim \operatorname{Normal}(0,~10)$",
                r"$\text{sigma} \sim \operatorname{HalfNormal}(0,~1)$",
                r"$\text{mu} \sim \operatorname{Deterministic}(f(\text{beta},~\text{alpha}))$",
                r"$\text{beta} \sim \operatorname{Normal}(0,~10)$",
                r"$\text{Z} \sim \operatorname{MvNormal}(f(),~f())$",
                r"$\text{nb_with_p_n} \sim \operatorname{NegBinom}(10,~\text{nbp})$",
                r"$\text{zip} \sim \operatorname{MarginalMixture}(f(),~\text{\$\operatorname{DiracDelta}(0)\$},~\text{\$\operatorname{Poisson}(5)\$})$",
                (
                    r"$\text{nested_mix} \sim \operatorname{MarginalMixture}(\text{<constant>},"
                    r"~\text{\$\operatorname{MarginalMixture}(f(),~\text{\\$\operatorname{DiracDelta}(0)\\$},~\text{\\$\operatorname{Poisson}(5)\\$})\$},"
                    r"~\text{\$\operatorname{Censored}(\text{\\$\operatorname{Bern}(0.5)\\$},~-1,~1)\$})$"
                ),
                r"$\text{Y_obs} \sim \operatorname{Normal}(\text{mu},~\text{sigma})$",
                r"$\text{pot} \sim \operatorname{Potential}(f(\text{beta},~\text{alpha}))$",
            ],
            ("latex", False): [
                r"$\text{alpha} \sim \operatorname{Normal}$",
                r"$\text{sigma} \sim \operatorname{HalfNormal}$",
                r"$\text{mu} \sim \operatorname{Deterministic}$",
                r"$\text{beta} \sim \operatorname{Normal}$",
                r"$\text{Z} \sim \operatorname{MvNormal}$",
                r"$\text{nb_with_p_n} \sim \operatorname{NegBinom}$",
                r"$\text{zip} \sim \operatorname{MarginalMixture}$",
                r"$\text{nested_mix} \sim \operatorname{MarginalMixture}$",
                r"$\text{Y_obs} \sim \operatorname{Normal}$",
                r"$\text{pot} \sim \operatorname{Potential}$",
            ],
        }

    def test__repr_latex_(self):
        for distribution, tex in zip(self.distributions, self.expected[("latex", True)]):
            assert distribution._repr_latex_() == tex

        model_tex = self.model._repr_latex_()

        # make sure each variable is in the model
        for tex in self.expected[("latex", True)]:
            for segment in tex.strip("$").split(r"\sim"):
                assert segment in model_tex

    def test_str_repr(self):
        for str_format in self.formats:
            for dist, text in zip(self.distributions, self.expected[str_format]):
                assert dist.str_repr(*str_format) == text

            model_text = self.model.str_repr(*str_format)
            for text in self.expected[str_format]:
                if str_format[0] == "latex":
                    for segment in text.strip("$").split(r"\sim"):
                        assert segment in model_text
                else:
                    assert text in model_text
