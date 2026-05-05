# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## PyMC v6 & PyTensor v3: ecosystem updates
#
# PyMC has been under steady development since the early 2010s.
# To mark the new major releases of PyMC v6 and PyTensor v3,
# we want to highlight the developments across the PyMC ecosystem
# (and its close cousin, the ArviZ ecosystem) that we're most excited about.
#
# ![image.png](attachment:b1fd1929-b4a9-4d66-8405-a405c7a33165.png)
#
# ### What's new in PyMC v6
# PyMC v6 starts faster, runs better, and can finally be **pip install**ed. No extra system setup required.
#
# Under the hood, we've streamlined the computational backend (PyTensor) and switched the default computational backend to Numba,
# while keeping it easy to opt into others on demand (C, JAX, MLX, …).
#
# For inference, PyMC now uses the nutpie NUTS sampler by default when installed, and pymc-extras provides easy access to Pathfinder and DADVI variational inference.
#
# On the modeling side, the new pymc.dims module makes it natural to define dimension-aware models,
# while sister packages add automatic marginalization, a clean state-space model API, and Bayesian Additive Regression Trees (BART).
#
# The surrounding ecosystem has grown and matured too.
# ArviZ has entered version 1.0, with a more modular and flexible design to support the rest of the Bayesian workflow.
# Tools like PyMC-BART and PreliZ are more tightly integrated,
# and new additions like Kulprit bring automated variable selection into the workflow.
#
#
# ### At a glance
# * Performance & usability: Numba as the new default backend, simple backend switching, clean pip install
# * Inference: nutpie NUTS sampler, Pathfinder and DADVI variational inference
# * Modeling: named dimensions (pymc.dims), automatic marginalization, state-space models, BART,
# * Building blocks: automatic logp, wrap_jax
# * Workflow: ArviZ 1.0, variable selection with Kulprit, prior elicitation with PreliZ

# %% [markdown]
# ## Performance
#
# PyMC hasn't always been a breeze to use. It's written in Python, and its backend
# (Theano, and its successor PyTensor) had accrued some weight over the years.
#
# A lot of effort was devoted to speeding up all stages of work:
# from import times, to model compilation and sampling runtime.

# %% [markdown]
# ### Numba is the new default backend
#
# Historically, PyMC compiled model functions to **C**, with some support for CUDA. In v6 we
# switch the default linker to **Numba**. For most CPU workloads Numba matches or beats C,
# while being far easier to maintain and extend.
# It also unlocks a few things the old C path always struggled with:
#
# - **Native advanced indexing** — The kind of operations that show up all the time in hierarchical models.
#   C-support for indexing operations has always been a source of friction.
#   Numba provides native support to (almost) all cases of advanced indexing with noticeably better performance.
#   Ongoing work to combine mathematical and indexing operations into fused kernels should yield further improvements.
# - **LAPACK bindings for linear algebra** — `cholesky`, `solve`, `eigh`, and friends aren't bottlenecked by Python.
# - **Fast `scan`** — historically the slowest piece of the C backend, `scan` has been rewired for Numba
#   and is now far closer to looping in a low-level language (with autodiff).
#   This should greatly aid time-series and other models with recursive structures.
# - **Native sparse support** — PyTensor supports sparse tensors operations in Numba.
#   Useful for spatial models (ICAR) and State-Space models, as well as inference routines like INLA.
#
#
# A knock-on win: **PyMC is finally `pip install`-safe.** The old C backend needed a working
# BLAS install *and* a C++ compiler — pip wheels bring neither. The first-time experience on a
# clean machine used to greet you with a prominent "PyTensor could not link to a BLAS
# installation" warning and degraded performance (or worse if a compiler couldn't be detected).
# Conda/mamba was the only reliable path.

# %% [markdown]
# ### One-liner backend selection with `backend=`
#
# You can pass `backend=` directly to **every** sampling entry point — `pm.sample`,
# `pm.sample_prior_predictive`, `pm.sample_posterior_predictive`, the deterministic-
# recomputation helpers, `pm.fit` (variational), `pm.sample_smc`, and the `log_density` /
# `logp` utilities.
#
# Accepted values include `"numba"`, `"c"`, and `"jax"`.
#
# You don't have to pass it at all — `backend="numba"` is the default, so `pm.sample(...)`
# already uses Numba. You only need the keyword when you want something else.
#
# **Availability:** `Numba` and C ship with PyMC — though C needs both a C++ compiler
# and a working BLAS install at runtime to be usable at all. JAX has to be installed separately.

# %%
import os
import warnings

# Silence JAX's "os.fork() was called" RuntimeWarning emitted by worker startup.
# Set both an in-process filter and PYTHONWARNINGS so spawned workers inherit it.
os.environ["PYTHONWARNINGS"] = "ignore:os.fork:RuntimeWarning"
warnings.filterwarnings("ignore", message=r"os\.fork\(\) was called.*", category=RuntimeWarning)

# %%
import numpy as np
import pymc as pm

rng = np.random.default_rng(0)
y_obs = rng.normal(1.0, 0.5, size=50)

with pm.Model() as simple_model:
    mu = pm.Normal("mu")
    sigma = pm.HalfNormal("sigma")
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

# %%
with simple_model:
    idata_numba = pm.sample(backend="numba", quiet=True)
    idata_jax = pm.sample(backend="jax", quiet=True)
    idata_c = pm.sample(backend="c", quiet=True)

# %% [markdown]
# All should arrive at the same place, if not at the same time.

# %%
print("numba mu:", float(idata_numba.posterior["mu"].mean()))
print("jax mu:  ", float(idata_jax.posterior["mu"].mean()))
print("c mu:    ", float(idata_c.posterior["mu"].mean()))

# %% [markdown]
# ## Inference

# %% [markdown]
# ### Nutpie: the new default NUTS sampler
#
# PyMC has historically latched onto Stan's implementation of NUTS,
# borrowing its code almost line-by-line.
# It was time to contribute something back, and that something is
# [**nutpie**](https://pymc-devs.github.io/nutpie/).
#
# You can easily install pymc and nutpie together as:
#
# ```bash
# pip install pymc[nutpie]
# ```
#
# Once installed, nutpie becomes the **default** NUTS in PyMC.
#
# Three things make it a step change over what we had before:
#
# - **Faster adaptation.** Nutpie often gets away with ~300 tuning draws where the old default
#   conservatively used 1000, and after tuning it takes fewer leapfrog steps per draw, with no
#   loss of accuracy. If you want to dive into the nitty-gritty, check out [Seyboldt, Carlson, & Carpenter (2026)](https://arxiv.org/abs/2603.18845).
# - **Low-rank mass-matrix adaptation.** For posteriors with strongly correlated parameters, the
#   diagonal mass matrix used by most HMC implementations is a poor fit. Nutpie offers a low-rank
#   extension of the diagonal that can dramatically reduce the number of gradient evaluations per
#   effective draw on funnels, regressions with collinear predictors, and similar shapes.
# - **Implemented in rust** with direct integration with Numba and JAX backends.
#
# An experimental **normalizing-flow adaptation** is also available for really difficult
# posteriors; see the [nutpie docs](https://pymc-devs.github.io/nutpie/nf-adapt.html).
#
# To be sure not to miss the train, set `nuts_sampler="nutpie"` explicitly.

# %%
with simple_model:
    idata_nutpie = pm.sample(nuts_sampler="nutpie")

# %%
print("nutpie mu:", float(idata_nutpie.posterior["mu"].mean()))

# %% [markdown]
# A classic nasty posterior: a simple linear regression where the covariate sits far from zero,
# making the intercept and slope strongly anti-correlated in the likelihood.

# %%
rng_corr = np.random.default_rng(2)
N = 200
x_far = rng_corr.normal(1000.0, 1.0, size=N)
y_far = 2.0 + 3.0 * x_far + rng_corr.normal(0.0, 1.0, size=N)

with pm.Model() as regression_far:
    a = pm.Normal("a", 0.0, 1e4)
    b = pm.Normal("b", 0.0, 10.0)
    sigma = pm.HalfNormal("sigma", 1.0)
    pm.Normal("y", mu=a + b * x_far, sigma=sigma, observed=y_far)

# %%
with regression_far:
    idata_diag = pm.sample(nuts={"adaptation": "diag"}, quiet=True)

pm.stats.summary(idata_diag, var_names=["a", "b"], kind="diagnostics")

# %% [markdown]
# The low rank adaptation fares much better than the default diagonal mass matrix adaptation.

# %%
with regression_far:
    idata_lr = pm.sample(nuts={"adaptation": "low_rank"}, quiet=True)

pm.stats.summary(idata_lr, var_names=["a", "b"], kind="diagnostics")

# %% [markdown]
# ### Variational inference (pymc-extras): Pathfinder & DADVI
#
# Pymc-extras ships with two modern Variational Inference solutions.
#
# **Pathfinder** Pathfinder is a parallel quasi-Newton variational approximation:
# it runs L-BFGS from multiple starting points and combines the resulting trajectories using Pareto-smoothed importance sampling.
# It's fast, parallelizable, and useful both as a NUTS warm-start and as a standalone approximation. See [Zhang et al. (2022)](https://arxiv.org/abs/2108.03782).

# %%
from pymc_extras import fit_pathfinder

with simple_model:
    idata_pf = fit_pathfinder(display_summary=False)

print("pathfinder posterior mu:", float(idata_pf.posterior["mu"].mean()))

# %% [markdown]
# **DADVI** — *Deterministic* ADVI.
# nstead of stochastic optimization of the ELBO, it draws a fixed Monte Carlo sample upfront
# and hands the resulting deterministic objective to a second-order optimizer.
# Three things fall out of this: convergence is unambiguous (no more "is the ELBO done bouncing around?"),
# off-the-shelf optimizers like L-BFGS work directly, and DADVI supports linear-response covariance corrections
# that fix the variance underestimation that mean-field ADVI is famous for. See [Giordano, Ingram & Broderick (2024)](https://jmlr.org/papers/volume25/23-1015/23-1015.pdf).

# %%
from pymc_extras.inference import fit_dadvi

with simple_model:
    idata_dadvi = fit_dadvi()

print("dadvi posterior mu:", float(idata_dadvi.posterior["mu"].mean()))

# %% [markdown]
# ## Modelling

# %% [markdown]
# ## Named dims with `pymc.dims`
#
# PyMC's `dims=` argument has existed since 2020 — you've probably used it to attach string
# names to axes of a random variable. In v6 it grows up. The new `pymc.dims` module (backed by
# PyTensor's XTensor) lets you write **entire models** against named dimensions, with automatic
# broadcasting, no more `[:, None, :]` gymnastics, and no more guessing which axis is which.
#
# A compact example (adapted from the [pymc.dims core
# notebook](https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/dims_module.html) — see it for a full tour).

# %%
import pymc.dims as pmd

coords = {
    "participant": range(5),
    "trial": range(20),
    "item": range(3),
}
observed_response_np = np.ones((5, 20), dtype=int)

with pm.Model(coords=coords) as dmodel:
    observed_response = pmd.Data(
        "observed_response",
        observed_response_np,
        dims=("participant", "trial"),
    )

    participant_preference = pmd.ZeroSumNormal(
        "participant_preference",
        core_dims="item",
        dims=("participant", "item"),
    )
    time_effects = pmd.Normal("time_effects", dims=("item", "trial"))

    trial_preference = pmd.Deterministic(
        "trial_preference",
        participant_preference + time_effects,
    )

    response = pmd.Categorical(
        "response",
        p=pmd.math.softmax(trial_preference, dim="item"),
        core_dims="item",
        observed=observed_response,
    )

# %% [markdown]
# Notice what is **not** in that model: no `None`-indexing to create new axes, no right-align
# conventions to keep track of, no numerical `axis` argument.
#
# Also new: the `Model.table()` method renders a compact summary of every variable together
# with its expression and dimensions — a much more compact view than the good old `Model.to_graphviz()`.

# %%
dmodel.table()


# %% [markdown]
# ### Automatic marginalization (pymc-extras)
#
# Discrete latent variables are a pain to sample with gradient-based MCMC.
# Traditionally you'd have to sum them out by hand and rewrite your model as a marginal likelihood.
# `pymc_extras.marginalize` automates that process.

# %%
from pymc_extras import marginalize, recover_marginals

rng_mix = np.random.default_rng(3)
y_mix = rng_mix.normal(np.tile([-2, 2], reps=50))
mix_coords = {"component": ["left", "right"], "obs": range(len(y_mix))}

with pm.Model(coords=mix_coords) as mix_model:
    w = pm.Dirichlet("w", a=np.ones(2))
    mu = pm.Normal("mu", mu=[-3.0, 3.0], sigma=2.0, dims="component")
    sigma = pm.HalfNormal("sigma", dims="component")
    z = pm.Categorical("z", p=w, dims="obs")
    pm.Normal("y", mu=mu[z], sigma=sigma[z], observed=y_mix, dims="obs")

marginal_model = marginalize(mix_model, ["z"])

with marginal_model:
    idata_mix = pm.sample(quiet=True)

pm.stats.summary(idata_mix)

# %% [markdown]
# After sampling the marginalized model, you can still **recover draws for the discrete
# latents** with `recover_marginals` — the component assignments for each observation come back
# as proper posterior estimates.

# %%
with marginal_model:
    idata_mix = recover_marginals(idata_mix)

# %%
pm.stats.summary(idata_mix.isel(obs=slice(None, 6)), var_names=["z"])

# %% [markdown]
# ### Statespace (pymc-extras)
#
# `pymc_extras.statespace` offers Bayesian statespace models with an increasingly rich high-level API: 
# `BayesianETS`, `BayesianSARIMAX`, `BayesianVARMAX`, plus a `structural` package for
# composing custom models out of level, trend, seasonal, and cyclical components. 
# The module has been maturing for a few cycles; getting ready for the inevitable IPO with it's own ticker.

# %%
from pymc_extras.statespace import BayesianETS

# Toy series with level + small trend + noise
t = np.arange(60)
y_ts = 10.0 + 0.05 * t + rng.normal(0, 0.5, size=60)

ss_mod = BayesianETS(
    order=("A", "N", "N"),
    endog_names=["y"],
    stationary_initialization=True,
    verbose=False,
)
with pm.Model(coords=ss_mod.coords) as ets_pymc:
    pm.Normal("initial_level", mu=y_ts[0], sigma=10)
    pm.Beta("alpha", alpha=3, beta=1)
    pm.Exponential("sigma_state", lam=1)
    ss_mod.build_statespace_graph(data=y_ts[:, None])
    idata_ets = pm.sample()

print("ETS posterior keys:", list(idata_ets.posterior.data_vars)[:6])

# %% [markdown]
# ### PyMC-BART
#
# [PyMC-BART](https://www.pymc.io/projects/bart/) brings **Bayesian Additive Regression
# Trees** to PyMC: a non-parametric prior on functions, expressed as an ensemble of
# decision trees.
#
# Newer versions ship with new plotting utilities (partial dependence, variable importance)
# and direct variable-importance integration with **Kulprit** (see next section)
# — letting BART guide downstream variable selection on a parametric follow-up model.
# An experimental Rust tree sampler is also available.

# %%
import pymc_bart as pmb

rng_bart = np.random.default_rng(5)
n = 200
X_bart = rng_bart.normal(size=(n, 4))
# Only columns 0 and 2 actually drive the response
y_bart = np.sin(1.5 * X_bart[:, 0]) + 0.7 * X_bart[:, 2] ** 2 + rng_bart.normal(0, 0.3, n)

with pm.Model() as bart_model:
    mu = pmb.BART("mu", X_bart, y_bart, m=50)
    sigma = pm.HalfNormal("sigma", 1.0)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_bart)
    idata_bart = pm.sample(progressbar="combined")

# Variable importance — columns 0 and 2 should dominate.
vi = pmb.compute_variable_importance(idata_bart, bartrv=mu, X=X_bart)
print("variable importance:", np.round(vi["r2_mean"], 3))

# %% [markdown]
# ## More building blocks
# %% [markdown]
# ### Drop-in JAX with `pytensor.wrap_jax`
#
# `pytensor.wrap_jax` makes any JAX-jittable function a first-class citizen in a PyMC model.
# That means the whole JAX ecosystem (Equinox neural networks, Diffrax ODE solvers, Optax optimizers)
# composes directly with your priors and likelihoods. Train a Bayesian neural network.
# Put a prior on the parameters of an ODE. Mix and match.

# %%
import jax.numpy as jnp

from pytensor import wrap_jax

@wrap_jax
def forward(x, a, b):
    return jnp.tanh(a * x + b)

rng = np.random.default_rng(2)
x_wj = rng.normal(size=30)
y_wj = np.tanh(0.8 * x_wj + 0.1) + rng.normal(0, 0.1, size=30)

with pm.Model() as wrap_jax_model:
    a = pm.Normal("a")
    b = pm.Normal("b")
    mu_wj = forward(x_wj, a, b)
    pm.Normal("y", mu=mu_wj, observed=y_wj)

    idata_wj = pm.sample(backend="jax", progressbar="combined")

pm.stats.summary(idata_wj)

# %% [markdown]
# Sampling still runs through nutpie — `backend="jax"` just tells PyMC to compile the model through the JAX path,
# so the wrapped function stays native end-to-end.

# %% [markdown]
# ### Automatic probability & `CustomDist`
#
# Most PPLs let you define models by chaining random variables and derive the log-density automatically.
# What's less known is that PyMC can also derive log-densities for transformations of existing distributions:
# exponentiating, taking maxima, slicing — without you writing the math yourself.
#
# Here we build a log-Student-T as exp of a Student-T, and ask PyMC for its log-density at 0.5

# %%
log_student_t = pm.math.exp(pm.StudentT.dist(nu=4, mu=0, sigma=2))

print(f"logp(LogStudentT, 0.5): {pm.logp(log_student_t, 0.5).eval():.4f}")

# %% [markdown]
# The way to wire this into a model is with `pm.CustomDist`, which turns any `dist`-building
# function into a first-class PyMC distribution:


# %%
def log_student_t_dist(nu, mu, sigma, size):
    return pm.math.exp(pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma, size=size))


with pm.Model() as m:
    mu = pm.Normal("mu")
    sigma = pm.HalfNormal("sigma")
    y = pm.CustomDist("y", 4.0, mu, sigma, dist=log_student_t_dist)


# %% [markdown]
# ## Bayesian workflow

# %% [markdown]
# ### ArviZ 1.0
#
# PyMC v6 is compatible with the now grown up **ArviZ 1.0**.
# Most of the changes are invisible to end users.
# The objects you get back from `pm.sample` still resembles the good old `InferenceData`,
# and the usual az.plot_* and az.summary calls keep working. But under the hood, a lot has changed.
#
# **InferenceData → xarray.DataTree.** The InferenceData is replaced by xarray's new DataTree.
# Arbitrary nesting is now allowed, I/O is more flexible (any format xarray supports is automatically available), and existing netCDF/Zarr files keep working.
#
# **New plotting library.** arviz-plots ships with three backends at near-feature-parity: Matplotlib, Bokeh, and Plotly. Plots return a PlotCollection object built on grammar-of-graphics ideas, which makes faceting, aesthetic mappings, and custom compositions much cleaner. Browse some of the new plots at the [example gallery](https://python.arviz.org/projects/plots/en/latest/gallery/index.html).
#
# **New default credible interval probability** has changed from 0.94 to 0.89, to remind everyone that no single number can rule us all. Perhaps more importantly, the default interval kind is now equal-tailed ("eti") rather than highest-density ("hdi"), and WAIC has been removed in favor of PSIS-LOO-CV for model comparison.
#
# Oh and `plot_trace` is now `plot_trace_dist`...

# %%
pm.plots.plot_trace_dist(idata_numba)

# %% [markdown]
# ### PreliZ — prior elicitation
#
# [PreliZ](https://preliz.readthedocs.io/) helps with one of the trickiest parts of the Bayesian workflow: prior choice.
#
# Recent PreliZ work makes PyMC distributions and pymc-extras `Prior` objects first-class
# citizens — `maxent`, matching, and plotting all accept them directly, and `from_pymc` /
# `from_prior` convert them explicitly.
#
# `maxent` also accepts any summary statistic (mean, variance, mode, …) as a fixed constraint,
# and new `match_moments` / `match_quantiles` helpers build univariate distributions from moment or quantile targets.
#
# The docs now ship a full [Distributions Gallery](https://preliz.readthedocs.io/en/latest/gallery/gallery.html).

# %%
import preliz as pz

# "Find a Gamma with mass 0.9 between 1 and 10 — and pin the mean to 4."
(prior, _fig) = pz.maxent(pz.Gamma(), lower=1, upper=10, mass=0.90, fixed_stat=("mean", 4))

# %%
with pm.Model() as m:
    x = prior.to_pymc("x")
    idata = pm.sample(quiet=True)
pm.stats.summary(idata)

# %% [markdown]
# ### Kulprit — variable selection
#
# [Kulprit](https://kulprit.readthedocs.io/) tackles a problem that shows up in almost
# every real study: you measured many predictors, but the follow-up can only afford a
# few. *Which ones?*
#
# Kulprit takes a **Bambi** model containing all candidate variables and automatically
# generates and ranks a set of plausible submodels with fewer variables using
# [**projective predictive inference**](https://kulprit.readthedocs.io/en/latest/explanation.html).
# The projection is much faster than re-fitting every submodel, and more reliable than
# LOO-based model comparison when the number of candidates is large.

# %%
# import bambi as bmb
# import kulprit as kpt
# import pandas as pd

# rng_k = np.random.default_rng(6)
# n_k, p_k = 200, 6
# X_k = pd.DataFrame(rng_k.normal(size=(n_k, p_k)), columns=[f"x{i}" for i in range(p_k)])
# # Only x0 and x2 are informative
# X_k["y"] = 0.5 + 1.5 * X_k["x0"] + 0.8 * X_k["x2"] + rng_k.normal(0, 0.3, n_k)

# full_model = bmb.Model("y ~ " + " + ".join(f"x{i}" for i in range(p_k)), X_k)
# idata_full = full_model.fit(chains=2, tune=200, draws=200, progressbar=False)

# ppi = kpt.ProjectionPredictive(full_model, idata_full)
# ppi.search()
# print(ppi)

# %% [markdown]
# ## It has been a long ride
#
# ... something something

# %%
