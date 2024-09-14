#   Copyright 2024 The PyMC Developers
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

import numpy as np

from pytensor.tensor.random import normal

from pymc import Bernoulli, Censored, CustomDist, Gamma, HalfCauchy, Mixture, StudentT, Truncated
from pymc.distributions import (
    Dirichlet,
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
from pymc.pytensorf import floatX


class BaseTestStrAndLatexRepr:
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


class TestMonolith(BaseTestStrAndLatexRepr):
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
            w = Dirichlet("w", [1, 1])
            nested_mix = Mixture("nested_mix", w, [comp_1, comp_2])

            # Expected value of outcome
            mu = Deterministic("mu", floatX(alpha + dot(X, b)))

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

            # add a deterministic that depends on an unnamed random variable
            pred = Deterministic("pred", Normal.dist(0, 1))

        self.distributions = [alpha, sigma, mu, b, Z, nb2, zip, w, nested_mix, Y_obs, pot]
        self.deterministics_or_potentials = [mu, pot, pred]
        # tuples of (formatting, include_params)
        self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            ("plain", True): [
                r"alpha ~ Normal(0, 10)",
                r"sigma ~ HalfNormal(0, 1)",
                r"mu ~ Deterministic(f(beta, alpha))",
                r"beta ~ Normal(0, 10)",
                r"Z ~ MultivariateNormal(f(), f())",
                r"nb_with_p_n ~ NegativeBinomial(10, nbp)",
                r"zip ~ MarginalMixture(f(), DiracDelta(0), Poisson(5))",
                r"w ~ Dirichlet(<constant>)",
                (
                    r"nested_mix ~ MarginalMixture(w, "
                    r"MarginalMixture(f(), DiracDelta(0), Poisson(5)), "
                    r"Censored(Bernoulli(0.5), -1, 1))"
                ),
                r"Y_obs ~ Normal(mu, sigma)",
                r"pot ~ Potential(f(beta, alpha))",
                r"pred ~ Deterministic(f(<normal>))",
            ],
            ("plain", False): [
                r"alpha ~ Normal",
                r"sigma ~ HalfNormal",
                r"mu ~ Deterministic",
                r"beta ~ Normal",
                r"Z ~ MultivariateNormal",
                r"nb_with_p_n ~ NegativeBinomial",
                r"zip ~ MarginalMixture",
                r"w ~ Dirichlet",
                r"nested_mix ~ MarginalMixture",
                r"Y_obs ~ Normal",
                r"pot ~ Potential",
                r"pred ~ Deterministic",
            ],
            ("latex", True): [
                r"$\text{alpha} \sim \operatorname{Normal}(0,~10)$",
                r"$\text{sigma} \sim \operatorname{HalfNormal}(0,~1)$",
                r"$\text{mu} \sim \operatorname{Deterministic}(f(\text{beta},~\text{alpha}))$",
                r"$\text{beta} \sim \operatorname{Normal}(0,~10)$",
                r"$\text{Z} \sim \operatorname{MultivariateNormal}(f(),~f())$",
                r"$\text{nb\_with\_p\_n} \sim \operatorname{NegativeBinomial}(10,~\text{nbp})$",
                r"$\text{zip} \sim \operatorname{MarginalMixture}(f(),~\operatorname{DiracDelta}(0),~\operatorname{Poisson}(5))$",
                r"$\text{w} \sim \operatorname{Dirichlet}(\text{<constant>})$",
                (
                    r"$\text{nested\_mix} \sim \operatorname{MarginalMixture}(\text{w},"
                    r"~\operatorname{MarginalMixture}(f(),~\operatorname{DiracDelta}(0),~\operatorname{Poisson}(5)),"
                    r"~\operatorname{Censored}(\operatorname{Bernoulli}(0.5),~-1,~1))$"
                ),
                r"$\text{Y\_obs} \sim \operatorname{Normal}(\text{mu},~\text{sigma})$",
                r"$\text{pot} \sim \operatorname{Potential}(f(\text{beta},~\text{alpha}))$",
                r"$\text{pred} \sim \operatorname{Deterministic}(f(\text{<normal>}))",
            ],
            ("latex", False): [
                r"$\text{alpha} \sim \operatorname{Normal}$",
                r"$\text{sigma} \sim \operatorname{HalfNormal}$",
                r"$\text{mu} \sim \operatorname{Deterministic}$",
                r"$\text{beta} \sim \operatorname{Normal}$",
                r"$\text{Z} \sim \operatorname{MultivariateNormal}$",
                r"$\text{nb\_with\_p\_n} \sim \operatorname{NegativeBinomial}$",
                r"$\text{zip} \sim \operatorname{MarginalMixture}$",
                r"$\text{w} \sim \operatorname{Dirichlet}$",
                r"$\text{nested\_mix} \sim \operatorname{MarginalMixture}$",
                r"$\text{Y\_obs} \sim \operatorname{Normal}$",
                r"$\text{pot} \sim \operatorname{Potential}$",
                r"$\text{pred} \sim \operatorname{Deterministic}",
            ],
        }


class TestData(BaseTestStrAndLatexRepr):
    def setup_class(self):
        with Model() as self.model:
            import pymc as pm

            with pm.Model() as model:
                a = pm.Normal("a", pm.Data("a_data", (2,)))
                b = pm.Normal("b", pm.Data("b_data", (2, 3)))
                c = pm.Normal("c", pm.Data("c_data", (2,)))
                d = pm.Normal("d", pm.Data("d_data", (2, 3)))

        self.distributions = [a, b, c, d]
        # tuples of (formatting, include_params)
        self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            ("plain", True): [
                r"a ~ Normal(2, 1)",
                r"b ~ Normal(<shared>, 1)",
                r"c ~ Normal(2, 1)",
                r"d ~ Normal(<shared>, 1)",
            ],
            ("plain", False): [
                r"a ~ Normal",
                r"b ~ Normal",
                r"c ~ Normal",
                r"d ~ Normal",
            ],
            ("latex", True): [
                r"$\text{a} \sim \operatorname{Normal}(2,~1)$",
                r"$\text{b} \sim \operatorname{Normal}(\text{<shared>},~1)$",
                r"$\text{c} \sim \operatorname{Normal}(2,~1)$",
                r"$\text{d} \sim \operatorname{Normal}(\text{<shared>},~1)$",
            ],
            ("latex", False): [
                r"$\text{a} \sim \operatorname{Normal}$",
                r"$\text{b} \sim \operatorname{Normal}$",
                r"$\text{c} \sim \operatorname{Normal}$",
                r"$\text{d} \sim \operatorname{Normal}$",
            ],
        }


def test_model_latex_repr_three_levels_model():
    with Model() as censored_model:
        mu = Normal("mu", 0.0, 5.0)
        sigma = HalfCauchy("sigma", 2.5)
        normal_dist = Normal.dist(mu=mu, sigma=sigma)
        censored_normal = Censored(
            "censored_normal", normal_dist, lower=-2.0, upper=2.0, observed=[1, 0, 0.5]
        )

    latex_repr = censored_model.str_repr(formatting="latex")
    expected = [
        "$$",
        "\\begin{array}{rcl}",
        "\\text{mu} &\\sim & \\operatorname{Normal}(0,~5)\\\\\\text{sigma} &\\sim & "
        "\\operatorname{HalfCauchy}(0,~2.5)\\\\\\text{censored\\_normal} &\\sim & "
        "\\operatorname{Censored}(\\operatorname{Normal}(\\text{mu},~\\text{sigma}),~-2,~2)",
        "\\end{array}",
        "$$",
    ]
    assert [line.strip() for line in latex_repr.split("\n")] == expected


def test_model_latex_repr_mixture_model():
    with Model() as mix_model:
        w = Dirichlet("w", [1, 1])
        mix = Mixture("mix", w=w, comp_dists=[Normal.dist(0.0, 5.0), StudentT.dist(7.0)])

    latex_repr = mix_model.str_repr(formatting="latex")
    expected = [
        "$$",
        "\\begin{array}{rcl}",
        "\\text{w} &\\sim & "
        "\\operatorname{Dirichlet}(\\text{<constant>})\\\\\\text{mix} &\\sim & "
        "\\operatorname{MarginalMixture}(\\text{w},~\\operatorname{Normal}(0,~5),~\\operatorname{StudentT}(7,~0,~1))",
        "\\end{array}",
        "$$",
    ]
    assert [line.strip() for line in latex_repr.split("\n")] == expected


def test_model_repr_variables_without_monkey_patched_repr():
    """Test that model repr does not rely on individual variables having the str_repr method monkey patched."""
    x = normal(name="x")
    assert not hasattr(x, "str_repr")

    model = Model()
    model.register_rv(x, "x")

    str_repr = model.str_repr()
    assert str_repr == "x ~ Normal(0, 1)"


def test_truncated_repr():
    with Model() as model:
        x = Truncated("x", Gamma.dist(1, 1), lower=0, upper=20)

    str_repr = model.str_repr(include_params=False)
    assert str_repr == "x ~ TruncatedGamma"


def test_custom_dist_repr():
    with Model() as model:

        def dist(mu, size):
            return Normal.dist(mu, 1, size=size)

        def random(rng, mu, size):
            return rng.normal(mu, size=size)

        x = CustomDist("x", 0, dist=dist, class_name="CustomDistNormal")
        x = CustomDist("y", 0, random=random, class_name="CustomRandomNormal")

    str_repr = model.str_repr(include_params=False)
    assert str_repr == "\n".join(["x ~ CustomDistNormal", "y ~ CustomRandomNormal"])


class TestLatexRepr:
    @staticmethod
    def simple_model() -> Model:
        with Model() as simple_model:
            error = HalfNormal("error", 0.5)
            alpha_a = Normal("alpha_a", 0, 1)
            Normal("y", alpha_a, error)
        return simple_model

    def test_latex_escaped_underscore(self):
        """
        Ensures that all underscores in model variable names are properly escaped for LaTeX representation
        """
        model = self.simple_model()
        model_str = model.str_repr(formatting="latex")
        assert "\\_" in model_str
        assert "_" not in model_str.replace("\\_", "")
