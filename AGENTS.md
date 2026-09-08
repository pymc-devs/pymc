# PyMC

PyMC is a probabilistic programming library for Python. It allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods.

**Current version: PyMC 6.x** — requires PyTensor 3.x and ArviZ 1.x. Key defaults:
- Default backend: **numba** (via PyTensor 3). Use `pytensor.config.linker = "cvm"` to revert to C.
- Default NUTS sampler: **nutpie** when installed (`pip install pymc[nutpie]`). Nutpie defaults to 400 tune steps; PyMC's own NUTS defaults to 1000.
- Inference data object: **`xarray.DataTree`** (ArviZ 1.0 replaced `arviz.InferenceData`).

## Repository layout

- `pymc/` — library source
- `tests/` — pytest test suite; mirrors `pymc/` structure
- `docs/` — Sphinx documentation
- `pyproject.toml` — project metadata and dev dependencies

## Development

```bash
pip install -e ".[dev]"
pytest tests/                          # full suite
pytest tests/test_sampling.py -x -q   # single file, fail-fast
```

Pre-commit hooks run ruff (lint + format). Run manually: `pre-commit run --all-files`.

## PyMC 6 / ArviZ 1.0 API — what changed

When writing or reviewing code that touches sampling, inference data, or plotting, apply these rules:

### InferenceData → DataTree

| PyMC 5 (old) | PyMC 6 (current) |
|---|---|
| `import arviz as az; az.InferenceData(...)` | `import xarray as xr; xr.DataTree(...)` |
| `idata.extend(other)` | `idata.update(other)` |
| `az.from_netcdf(path)` | `az.from_netcdf(path)` *(still works)* |
| `idata.to_netcdf(path)` | `idata.to_netcdf(path)` *(still works)* |

### Plotting

| PyMC 5 (old) | PyMC 6 (current) |
|---|---|
| `az.plot_trace(idata)` | `az.plot_trace_dist(idata)` or `az.plot_rank_dist(idata)` (preferred) |

### Credible intervals

ArviZ 1.0 changed the default from **0.94 HDI** to **0.89 equal-tailed interval (ETI)**. When writing docs or examples, use `hdi_prob=0.94` explicitly if HDI is intended, or accept the new 0.89 ETI default.

### `sample_posterior_predictive` — new volatility API

PyMC 6 adds `sample_vars` and `freeze_vars` to control which variables are resampled:

```python
# Old: var_names controlled sampling semantics (confusing)
# New: var_names only controls what is saved; use sample_vars/freeze_vars explicitly
with model:
    pm.set_data({"x": x_new})
    ppc = pm.sample_posterior_predictive(
        idata,
        sample_vars=["y"],   # resample these
        freeze_vars=["mu"],  # reuse from trace as-is
    )
```

Variables with changed dependencies issue a **warning** (not automatic resampling). Suppress by assigning to `sample_vars` or `freeze_vars`.

### `sample_prior_predictive`

The deprecated `samples` argument was removed. Use `draws` instead:

```python
prior = pm.sample_prior_predictive(draws=500)  # not samples=500
```

## Bayesian modeling conventions (for examples and tests)

These apply when writing example notebooks, documentation, or model-based tests.

### Installation

```bash
# Recommended: install with nutpie for 2-5x faster sampling
pip install pymc[nutpie]
# Or via conda-forge (preferred for compiled backends)
mamba install -c conda-forge pymc nutpie arviz arviz-stats preliz
```

### Model template

```python
import pymc as pm
import arviz as az
import numpy as np

RANDOM_SEED = sum(map(ord, "analysis-name"))  # descriptive, reproducible
rng = np.random.default_rng(RANDOM_SEED)

coords = {"obs": obs_ids, "feature": feature_names}

with pm.Model(coords=coords) as model:
    # Data containers — required for out-of-sample prediction
    x = pm.Data("x", x_obs, dims=("obs", "feature"))

    # Priors — always document the justification
    beta = pm.Normal("beta", mu=0, sigma=1, dims="feature")
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu = pm.math.dot(x, beta)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")

    # Prior predictive check
    prior = pm.sample_prior_predictive(draws=500, random_seed=rng)

    # Inference — nutpie is default when installed
    idata = pm.sample(random_seed=rng)
    idata.update(prior)  # DataTree.update, not .extend

    # Posterior predictive check
    pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=rng)

    # Required for LOO-CV (nutpie does NOT store this automatically)
    pm.compute_log_likelihood(idata, model=model)

# Save immediately — before any post-processing
idata.to_netcdf("results.nc")
```

### Sampler

```python
# nutpie is the default when installed; explicit form:
idata = pm.sample(nuts_sampler="nutpie", random_seed=rng)

# nutpie cannot handle discrete parameters or the `ordered` transform.
# For those models, omit nuts_sampler (falls back to PyMC's NUTS):
idata = pm.sample(random_seed=rng)
```

Never hardcode `nuts_sampler` when writing generic utility code — let the user's environment pick the default.

### Seeds

Never use `random_seed=42`. Derive seeds from the analysis name so they're meaningful and reproducible:

```python
RANDOM_SEED = sum(map(ord, "my-model-name"))
rng = np.random.default_rng(RANDOM_SEED)
```

### Diagnostics — minimum checklist

Every model run must pass all four before results are reported:

```python
# 1. Single-call convergence report (requires arviz-stats >= 1.0.0)
from arviz_stats import diagnose
diagnose(idata)  # covers R-hat, ESS, divergences, tree depth, E-BFMI

# 2. Thresholds (strict)
assert idata.sample_stats["diverging"].sum() == 0       # zero divergences
# R-hat < 1.01 for all parameters (not 1.05, not 1.1 — those thresholds are outdated)
# ESS_bulk > 100 * n_chains and ESS_tail > 100 * n_chains

# 3. Posterior predictive check
az.plot_ppc(idata, kind="cumulative")
az.plot_loo_pit(idata, y="y_obs_var_name")  # calibration

# 4. LOO-CV for model comparison
loo = az.loo(idata, pointwise=True)
# Check: (loo.pareto_k > 0.7).sum() should be 0 or very small
```

### Hierarchical models — parameterization

Prefer non-centered for weak data, centered for strong data:

```python
# Non-centered (avoids funnel geometry, better for divergences)
alpha_offset = pm.Normal("alpha_offset", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset, dims="group")

# Centered (faster when data strongly constrains group-level params)
alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="group")
```

### Gaussian Processes — prefer HSGP

For n > 500 or 1–3D inputs, use HSGP (O(nm)) over exact GP (O(n³)):

```python
gp = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov)
f = gp.prior("f", X=X[:, None])  # X must be 2D
```

### Out-of-sample prediction

```python
with model:
    pm.set_data({"x": x_new})
    oos = pm.sample_posterior_predictive(idata, predictions=True, random_seed=rng)
```

### Prior sensitivity

```python
from arviz_stats import psense_summary
from arviz_plots import plot_psense_dist

# Requires log_likelihood and log_prior in idata
pm.compute_log_prior(idata, model=model)
psense_summary(idata)
plot_psense_dist(idata)
```

### Model inspection (new in PyMC 6)

```python
model.table()   # tabular summary of variables, shapes, distributions
model.debug()   # checks for common specification errors
model.point_logps()  # log-probability at initial values
```

### pymc-extras

```python
import pymc_extras as pmx

pmx.marginalize(model, ["discrete_var"])  # marginalize discrete params for NUTS
pmx.fit_laplace(model)                    # fast Laplace approximation
pmx.R2D2M2CP("r2d2", output_sigma=..., input_sigma=..., dims="features")
```

## Common pitfalls (do not repeat)

| Pitfall | Fix |
|---|---|
| `idata.extend(other)` | `idata.update(other)` — ArviZ 1.0 |
| `az.plot_trace(idata)` | `az.plot_trace_dist(idata)` — ArviZ 1.0 |
| `pm.sample_prior_predictive(samples=N)` | Use `draws=N` — `samples` removed |
| nutpie + `idata_kwargs={"log_likelihood": True}` | nutpie silently ignores this; call `pm.compute_log_likelihood(idata)` after sampling |
| `var_names` in `sample_posterior_predictive` to control sampling | Use `sample_vars`/`freeze_vars` — `var_names` now only controls saved variables |
| Hardcoded `random_seed=42` | Use `sum(map(ord, "model-name"))` |
| `np.median(posterior_probs)` | Use `np.mean` — median violates probability coherence |
| Python `if` in model body | Use `pm.math.switch` or `pt.where` |
| `az.loo(idata)` → `plot_khat(idata)` | Pass the LOO object: `plot_khat(loo)` |
| R-hat threshold of 1.05 or 1.1 | Threshold is **1.01** — the others are outdated |
| Flat priors on scale parameters (`HalfCauchy`, `HalfFlat`) | Use `Gamma(2, ...)` or `Exponential` to avoid funnels |

## Testing

- Unit tests go in `tests/` mirroring the source tree.
- Use `pytest.mark.slow` for tests that run MCMC; these are skipped in fast CI.
- Use `pm.sample(draws=10, tune=5, chains=1)` in tests — just enough to check shapes and that sampling runs.
- For InferenceData comparisons use `az.assert_close(idata_a, idata_b)`.
- Avoid saving NetCDF files in tests; use in-memory objects.

## Documentation

- Docstrings follow NumPy style.
- Example notebooks use marimo (`.py` format) unless an existing Jupyter notebook is being updated.
- All code examples must work with the current PyMC version on the main branch.
- Use `pm.Model(coords=coords)` with named dimensions in all new examples — never raw `shape=` arrays without coords.
