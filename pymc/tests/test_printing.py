#   Copyright 2023 The PyMC Developers
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
import pytensor.tensor as pt
import pytest

from pymc import Bernoulli, Censored, HalfCauchy, Mixture, StudentT
from pymc.data import ConstantData, MutableData
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
from pymc.printing import _is_potential_or_deterministic, str_for_model_var
from pymc.pytensorf import floatX


class BaseTestStrAndLatexRepr:
    @pytest.mark.parametrize("formatting", ["plain", "latex"])
    def test_var_str_repr(self, formatting):
        for dist, expected in zip(self.distributions, self.expected[formatting]):
            assert dist.str_repr(formatting) == expected

    @pytest.mark.parametrize("formatting", ["plain", "latex"])
    def test_model_str_repr(self, formatting):
        # Tests that each printed variable is in the model string, but
        # does not test for ordering
        model_str = self.model.str_repr(formatting)
        for expected in self.expected[formatting]:
            if formatting == "latex":
                expected = expected.replace(r"\sim", r"&\sim &").strip("$")
            assert expected in model_str

    def test_var_repr_latex(self):
        for dist in self.distributions:
            assert dist._repr_latex_() == dist.str_repr(formatting="latex")

    def test_model_repr_latex(self):
        assert self.model._repr_latex_() == self.model.str_repr(formatting="latex")


class TestMonolith(BaseTestStrAndLatexRepr):
    def setup_class(self):
        # True parameter values
        alpha0, sigma0, beta0 = 1, 1, [1, 2.5]
        # Size of dataset
        size = 100
        # Predictor variable
        X = np.random.normal(size=(size, 2)).dot(np.array([[1, 0], [0, 0.2]]))
        # Simulate outcome variable
        Y = alpha0 + X.dot(beta0) + np.random.randn(size) * sigma0
        with Model() as self.model:
            # TODO: some variables commented out here as they're not working properly
            # in v4 yet (9-jul-2021), so doesn't make sense to test str/latex for them

            # Priors for unknown model parameters
            alpha = Normal("alpha", mu=0, sigma=10)
            # TODO why is this observed?
            beta = Normal("beta", mu=0, sigma=10, size=(2,), observed=beta0)
            sigma = HalfNormal("sigma", sigma=1)

            # Test Cholesky parameterization
            Z = MvNormal("Z", mu=np.zeros(2), chol=np.eye(2), size=(2,))

            # NegativeBinomial representations to test issue 4186
            # nb1 = pm.NegativeBinomial(
            #     "nb_with_mu_alpha", mu=pm.Normal("nbmu"), alpha=pm.Gamma("nbalpha", mu=6, sigma=1)
            # )
            nbp = Uniform("nbp")
            nb_with_p_n = NegativeBinomial("nb_with_p_n", p=nbp, n=10)

            # SymbolicRV
            zip = ZeroInflatedPoisson("zip", 0.5, 5)

            # Nested SymbolicRV
            comp_1 = ZeroInflatedPoisson.dist(0.5, 5)
            comp_2 = Censored.dist(Bernoulli.dist(0.5), -1, 1)
            w = Dirichlet("w", [1, 1])
            nested_mix = Mixture("nested_mix", w, [comp_1, comp_2])

            # Expected value of outcome
            mu = Deterministic("mu", floatX(alpha + dot(X, beta)))

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

        self.distributions = [
            alpha,
            sigma,
            Z,
            nbp,
            nb_with_p_n,
            zip,
            w,
            nested_mix,
            kron_normal,
            dm,
            mu,
            beta,
            Y_obs,
            pot,
        ]
        # tuples of (formatting, include_params)
        # self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            "plain": [
                "alpha ~ N(0, 10)",
                "sigma ~ N**+(0, 1)",
                "Z ~ N(f(), f())",
                "nbp ~ U(0, 1)",
                "nb_with_p_n ~ NB(10, nbp)",
                "zip ~ MarginalMixture(f(), DiracDelta(0), Pois(5))",
                "w ~ Dir(<constant (2,)>)",
                "nested_mix ~ MarginalMixture(w, MarginalMixture(f(), DiracDelta(0), Pois(5)), Censored(Bern(0.5), -1, 1))",
                "kron_normal ~ KroneckerNormal(<constant (12,)>, 0, <constant (3, 3)>, <constant (4, 4)>)",
                "dm ~ DirichletMN(5, <constant (3,)>)",
                "mu ~ Deterministic(f(beta, alpha))",
                "beta ~ N(0, 10)",
                "Y_obs ~ N(mu, sigma)",
                "pot ~ Potential(f(beta, alpha))",
            ],
            # ("plain", False): [
            #     r"alpha ~ N",
            #     r"sigma ~ N**+",
            #     r"mu ~ Deterministic",
            #     r"beta ~ N",
            #     r"Z ~ N",
            #     r"nb_with_p_n ~ NB",
            #     r"zip ~ MarginalMixture",
            #     r"w ~ Dir",
            #     r"nested_mix ~ MarginalMixture",
            #     r"Y_obs ~ N",
            #     r"pot ~ Potential",
            # ],
            "latex": [
                r"$\text{alpha} \sim \operatorname{N}(0,~10)$",
                r"$\text{sigma} \sim \operatorname{N^{+}}(0,~1)$",
                r"$\text{Z} \sim \operatorname{N}(f(),~f())$",
                r"$\text{nbp} \sim \operatorname{U}(0,~1)$",
                r"$\text{nb}\_\text{with}\_\text{p}\_\text{n} \sim \operatorname{NB}(10,~\text{nbp})$",
                r"$\text{zip} \sim \operatorname{MarginalMixture}(f(),~\operatorname{DiracDelta}(0),~\operatorname{Pois}(5))$",
                r"$\text{w} \sim \operatorname{Dir}(\text{<constant (2,)>})$",
                r"$\text{nested}\_\text{mix} \sim \operatorname{MarginalMixture}(\text{w},~\operatorname{MarginalMixture}(f(),~\operatorname{DiracDelta}(0),~\operatorname{Pois}(5)),~\operatorname{Censored}(\operatorname{Bern}(0.5),~\text{-1},~1))$",
                r"$\text{kron}\_\text{normal} \sim \operatorname{KroneckerNormal}(\text{<constant (12,)>},~0,~\text{<constant (3, 3)>},~\text{<constant (4, 4)>})$",
                r"$\text{dm} \sim \operatorname{DirichletMN}(5,~\text{<constant (3,)>})$",
                r"$\text{mu} \sim \operatorname{Deterministic}(f(\text{beta},~\text{alpha}))$",
                r"$\text{beta} \sim \operatorname{N}(0,~10)$",
                r"$\text{Y}\_\text{obs} \sim \operatorname{N}(\text{mu},~\text{sigma})$",
                r"$\text{pot} \sim \operatorname{Potential}(f(\text{beta},~\text{alpha}))$",
            ],
            # ("latex", False): [
            #     r"$\text{alpha} \sim \operatorname{N}$",
            #     r"$\text{sigma} \sim \operatorname{N^{+}}$",
            #     r"$\text{mu} \sim \operatorname{Deterministic}$",
            #     r"$\text{beta} \sim \operatorname{N}$",
            #     r"$\text{Z} \sim \operatorname{N}$",
            #     r"$\text{nb_with_p_n} \sim \operatorname{NB}$",
            #     r"$\text{zip} \sim \operatorname{MarginalMixture}$",
            #     r"$\text{w} \sim \operatorname{Dir}$",
            #     r"$\text{nested_mix} \sim \operatorname{MarginalMixture}$",
            #     r"$\text{Y_obs} \sim \operatorname{N}$",
            #     r"$\text{pot} \sim \operatorname{Potential}$",
            # ],
        }


class TestData(BaseTestStrAndLatexRepr):
    def setup_class(self):
        with Model() as self.model:
            a = Normal("a", MutableData("a_data", (2,)))
            b = Normal("b", MutableData("b_data", (2, 3)))
            c = Normal("c", ConstantData("c_data", (2,)))
            d = Normal("d", ConstantData("d_data", (2, 3)))

        self.distributions = [a, b, c, d]
        # tuples of (formatting, include_params)
        # self.formats = [("plain", True), ("plain", False), ("latex", True), ("latex", False)]
        self.expected = {
            "plain": [
                r"a ~ N(2, 1)",
                r"b ~ N(<shared (2,)>, 1)",
                r"c ~ N(2, 1)",
                r"d ~ N(<constant (2,)>, 1)",
            ],
            # ("plain", False): [
            #     r"a ~ N",
            #     r"b ~ N",
            #     r"c ~ N",
            #     r"d ~ N",
            # ],
            "latex": [
                r"$\text{a} \sim \operatorname{N}(2,~1)$",
                r"$\text{b} \sim \operatorname{N}(\text{<shared (2,)>},~1)$",
                r"$\text{c} \sim \operatorname{N}(2,~1)$",
                r"$\text{d} \sim \operatorname{N}(\text{<constant (2,)>},~1)$",
            ],
            # ("latex", False): [
            #     r"$\text{a} \sim \operatorname{N}$",
            #     r"$\text{b} \sim \operatorname{N}$",
            #     r"$\text{c} \sim \operatorname{N}$",
            #     r"$\text{d} \sim \operatorname{N}$",
            # ],
        }


class TestNestedModel(BaseTestStrAndLatexRepr):
    def setup_class(self):
        with Model("m1") as self.model:
            n1 = Normal("n1")
            with Model("m2") as self.m2:
                n2 = Normal("n2")
        self.distributions = [n1, n2]
        self.expected = {
            "plain": ["m1::n1 ~ N(0, 1)", "m1::m2::n2 ~ N(0, 1)"],
            "latex": [
                r"$\text{m1::n1} \sim \operatorname{N}(0,~1)$",
                r"$\text{m1::m2::n2} \sim \operatorname{N}(0,~1)$",
            ],
        }

    def test_nested_model_str_repr_plain(self):
        assert self.m2.str_repr(formatting="plain") == self.expected["plain"][-1]


class TestEdgeCharacters(BaseTestStrAndLatexRepr):
    def setup_class(self):
        with Model("model_with_under$cores__and~stuff") as self.model:
            n = Normal("n")
            u_dollar = Uniform("u$")
            hn_tilde = HalfNormal("h~n")
            n_offset = Deterministic("n_offset", n + 1)
            d = Deterministic("d", n_offset + u_dollar + hn_tilde)
        self.distributions = [n, u_dollar, hn_tilde, n_offset, d]
        self.expected = {
            "plain": [
                "model_with_under$cores__and-stuff::n ~ N(0, 1)",
                "model_with_under$cores__and-stuff::u$ ~ U(0, 1)",
                "model_with_under$cores__and-stuff::h-n ~ N**+(0, 1)",
                "model_with_under$cores__and-stuff::n_offset ~ Deterministic(f(model_with_under$cores__and-stuff::n))",
                "model_with_under$cores__and-stuff::d ~ Deterministic(f(model_with_under$cores__and-stuff::h-n, model_with_under$cores__and-stuff::u$, model_with_under$cores__and-stuff::n))",
            ],
            "latex": [
                r"$\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::n} \sim \operatorname{N}(0,~1)$",
                r"$\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::u\$} \sim \operatorname{U}(0,~1)$",
                r"$\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::h-n} \sim \operatorname{N^{+}}(0,~1)$",
                r"$\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::n}\_\text{offset} \sim \operatorname{Deterministic}(f(\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::n}))$",
                r"$\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::d} \sim \operatorname{Deterministic}(f(\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::h-n},~\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::u\$},~\text{model}\_\text{with}\_\text{under\$cores}\_\text{}\_\text{and-stuff::n}))$",
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
        r"\begin{array}{rcl}",
        r"\text{mu} &\sim & \operatorname{N}(0,~5)\\\text{sigma} &\sim & "
        r"\operatorname{C^{+}}(0,~2.5)\\\text{censored}\_\text{normal} &\sim & "
        r"\operatorname{Censored}(\operatorname{N}(\text{mu},~\text{sigma}),~\text{-2},~2)",
        r"\end{array}",
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
        r"\begin{array}{rcl}",
        r"\text{w} &\sim & \operatorname{Dir}(\text{<constant (2,)>})\\\text{mix} &\sim & "
        r"\operatorname{MarginalMixture}(\text{w},~\operatorname{N}(0,~5),~\operatorname{StudentT}(7,~0,~1))",
        r"\end{array}",
        "$$",
    ]
    assert [line.strip() for line in latex_repr.split("\n")] == expected


def test_potential_input_no_owner():
    with Model():
        p = Potential("p", pt.constant(1))
    assert p.str_repr(formatting="plain") == "p ~ Potential(f())"
    assert p.str_repr(formatting="latex") == "$\\text{p} \\sim \\operatorname{Potential}(f())$"


def test_var_unnamed():
    v = Normal.dist()
    assert str_for_model_var(v, formatting="plain") == "<unnamed> ~ N(0, 1)"
    assert (
        str_for_model_var(v, formatting="latex")
        == "$\\text{<unnamed>} \\sim \\operatorname{N}(0,~1)$"
    )


def test_unsupported_var():
    """Not a model variable or dist() output"""
    with pytest.raises(ValueError, match="must be a model variable or the output of .dist()"):
        str_for_model_var(pt.constant(1), dist_name="_")


def test_unsupported_input_var():
    """Model variable has bad input"""
    with Model():
        # This fails in model_to_graphviz() due to MissingInputError But using
        # pt.scalar() rather than vector works in graphviz, but not pretty-print.
        x = Normal("x", mu=pt.vector())
    with pytest.raises(ValueError, match="Unidentified variable in dist or expression args"):
        x.str_repr()


def test_missing_dist_name():
    _print_name_temp = Normal.rv_op._print_name
    try:
        with Model() as m:
            n = Normal("n")
        n.owner.op._print_name = None  # delattr fails for some reason
        with pytest.raises(ValueError, match="Missing distribution name"):
            n.str_repr()
        with pytest.raises(ValueError, match="Missing distribution name"):
            m.str_repr()
    finally:
        Normal.rv_op._print_name = _print_name_temp


def test_include_params_warning():
    with Model() as m:
        n = Normal("n")
    with pytest.warns(FutureWarning, match="`include_params` argument has been deprecated"):
        n.str_repr(include_params="True")
    with pytest.warns(FutureWarning, match="`include_params` argument has been deprecated"):
        m.str_repr(include_params="True")


@pytest.mark.parametrize("formatting", ["plainblah", "latexblah", "aaa"])
def test_unsupported_formatting(formatting):
    with Model() as m:
        n = Normal("n")
    with pytest.raises(ValueError, match="Formatting method not recognized"):
        n.str_repr(formatting=formatting)
    with pytest.raises(ValueError, match="Formatting method not recognized"):
        m.str_repr(formatting=formatting)


def test_is_potential_or_deterministic():
    # Ensure printing.py is kept in line with model.py
    with Model():
        n = Normal("n")
        x = ConstantData("data", 1)
        p = Potential("p", x + 1)
        d = Deterministic("d", x + 1)
    assert not _is_potential_or_deterministic(n)
    assert not _is_potential_or_deterministic(x)
    assert _is_potential_or_deterministic(p)
    assert _is_potential_or_deterministic(d)
