# Release Notes

## PyMC3 3.11.0 (21 January 2021)

This release breaks some APIs w.r.t. `3.10.0`. It also brings some dreadfully awaited fixes, so be sure to go through the (breaking) changes below.

### Breaking Changes
- Python 3.6 support was dropped (by no longer testing) and Python 3.9 was added (see [#4332](https://github.com/pymc-devs/pymc3/pull/4332)).
- Changed shape behavior: __No longer collapse length 1 vector shape into scalars.__ (see [#4206](https://github.com/pymc-devs/pymc3/issue/4206) and [#4214](https://github.com/pymc-devs/pymc3/pull/4214))
  - __Applies to random variables and also the `.random(size=...)` kwarg!__
  - To create scalar variables you must now use `shape=None` or `shape=()`.
  - __`shape=(1,)` and `shape=1` now become vectors.__ Previously they were collapsed into scalars
  - 0-length dimensions are now ruled illegal for random variables and raise a `ValueError`.
- In `sample_prior_predictive` the `vars` kwarg was removed in favor of `var_names` (see [#4327](https://github.com/pymc-devs/pymc3/pull/4327)).
- Removed `theanof.set_theano_config` because it illegally changed Theano's internal state (see [#4329](https://github.com/pymc-devs/pymc3/pull/4329)).
- We now depend on `Theano-PyMC` version `1.1.0` exactly (see [#4405](https://github.com/pymc-devs/pymc3/pull/4405)). Major refactorings were done in `Theano-PyMC` 1.1.0. If you implement custom `Op`s or interact with Theano in any way yourself, make sure to read the [Theano-PyMC 1.1.0 release notes](https://github.com/pymc-devs/Theano-PyMC/releases/tag/rel-1.1.0).

### New Features
- Option to set `check_bounds=False` when instantiating `pymc3.Model()`. This turns off bounds checks that ensure that input parameters of distributions are valid. For correctly specified models, this is unneccessary as all parameters get automatically transformed so that all values are valid. Turning this off should lead to faster sampling (see [#4377](https://github.com/pymc-devs/pymc3/pull/4377)).
- `OrderedProbit` distribution added (see [#4232](https://github.com/pymc-devs/pymc3/pull/4232)).
- `plot_posterior_predictive_glm` now works with `arviz.InferenceData` as well (see [#4234](https://github.com/pymc-devs/pymc3/pull/4234))
- Add `logcdf` method to all univariate discrete distributions (see [#4387](https://github.com/pymc-devs/pymc3/pull/4387)).
- Add `random` method to `MvGaussianRandomWalk` (see [#4388](https://github.com/pymc-devs/pymc3/pull/4388))
- `AsymmetricLaplace` distribution added (see [#4392](https://github.com/pymc-devs/pymc3/pull/4392)).
- `DirichletMultinomial` distribution added (see [#4373](https://github.com/pymc-devs/pymc3/pull/4373)).
- Added a new `predict` method to `BART` to compute out of sample predictions (see [#4310](https://github.com/pymc-devs/pymc3/pull/4310)).

### Maintenance
- Fixed bug whereby partial traces returns after keyboard interrupt during parallel sampling had fewer draws than would've been available [#4318](https://github.com/pymc-devs/pymc3/pull/4318)
- Make `sample_shape` same across all contexts in `draw_values` (see [#4305](https://github.com/pymc-devs/pymc3/pull/4305)).
- The notebook gallery has been moved to https://github.com/pymc-devs/pymc-examples (see [#4348](https://github.com/pymc-devs/pymc3/pull/4348)).
- `math.logsumexp` now matches `scipy.special.logsumexp` when arrays contain infinite values (see [#4360](https://github.com/pymc-devs/pymc3/pull/4360)).
- Fixed mathematical formulation in `MvStudentT` random method. (see [#4359](https://github.com/pymc-devs/pymc3/pull/4359))
- Fix issue in `logp` method of `HyperGeometric`. It now returns `-inf` for invalid parameters (see [4367](https://github.com/pymc-devs/pymc3/pull/4367))
- Fixed `MatrixNormal` random method to work with parameters as random variables. (see [#4368](https://github.com/pymc-devs/pymc3/pull/4368))
- Update the `logcdf` method of several continuous distributions to return -inf for invalid parameters and values, and raise an informative error when multiple values cannot be evaluated in a single call. (see [4393](https://github.com/pymc-devs/pymc3/pull/4393) and [#4421](https://github.com/pymc-devs/pymc3/pull/4421))
- Improve numerical stability in `logp` and `logcdf` methods of `ExGaussian` (see [#4407](https://github.com/pymc-devs/pymc3/pull/4407))
- Issue UserWarning when doing prior or posterior predictive sampling with models containing Potential factors (see [#4419](https://github.com/pymc-devs/pymc3/pull/4419))
- Dirichlet distribution's `random` method is now optimized and gives outputs in correct shape (see [#4416](https://github.com/pymc-devs/pymc3/pull/4407))
- Attempting to sample a named model with SMC will now raise a `NotImplementedError`. (see [#4365](https://github.com/pymc-devs/pymc3/pull/4365))

**Release manager** for 3.11.0: Eelke Spaak ([@Spaak](https://github.com/Spaak))

## PyMC3 3.10.0 (7 December 2020)

This is a major release with many exciting new features. The biggest change is that we now rely on our own fork of [Theano-PyMC](https://github.com/pymc-devs/Theano-PyMC). This is in line with our [big announcement about our commitment to PyMC3 and Theano](https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b).

When upgrading, make sure that `Theano-PyMC` and not `Theano` are installed (the imports remain unchanged, however). If not, you can uninstall `Theano`:
```
conda remove theano
```

And to install:
```
conda install -c conda-forge theano-pymc
```

Or, if you are using pip (not recommended):
```
pip uninstall theano
```
And to install:
```
pip install theano-pymc
```

This new version of `Theano-PyMC` comes with an experimental JAX backend which, when combined with the new and experimental JAX samplers in PyMC3, can greatly speed up sampling in your model. As this is still very new, please do not use it in production yet but do test it out and let us know if anything breaks and what results you are seeing, especially speed-wise.

### New features
- New experimental JAX samplers in `pymc3.sample_jax` (see [notebook](https://docs.pymc.io/notebooks/GLM-hierarchical-jax.html) and [#4247](https://github.com/pymc-devs/pymc3/pull/4247)). Requires JAX and either TFP or numpyro.
- Add MLDA, a new stepper for multilevel sampling. MLDA can be used when a hierarchy of approximate posteriors of varying accuracy is available, offering improved sampling efficiency especially in high-dimensional problems and/or where gradients are not available (see [#3926](https://github.com/pymc-devs/pymc3/pull/3926))
- Add Bayesian Additive Regression Trees (BARTs) [#4183](https://github.com/pymc-devs/pymc3/pull/4183))
- Added `pymc3.gp.cov.Circular` kernel for Gaussian Processes on circular domains, e.g. the unit circle (see [#4082](https://github.com/pymc-devs/pymc3/pull/4082)).
- Added a new `MixtureSameFamily` distribution to handle mixtures of arbitrary dimensions in vectorized form for improved speed (see [#4185](https://github.com/pymc-devs/pymc3/issues/4185)).
- `sample_posterior_predictive_w` can now feed on `xarray.Dataset` - e.g. from `InferenceData.posterior`. (see [#4042](https://github.com/pymc-devs/pymc3/pull/4042))
- Change SMC metropolis kernel to independent metropolis kernel [#4115](https://github.com/pymc-devs/pymc3/pull/4115))
- Add alternative parametrization to NegativeBinomial distribution in terms of n and p (see [#4126](https://github.com/pymc-devs/pymc3/issues/4126))
- Added semantically meaningful `str` representations to PyMC3 objects for console, notebook, and GraphViz use (see [#4076](https://github.com/pymc-devs/pymc3/pull/4076), [#4065](https://github.com/pymc-devs/pymc3/pull/4065), [#4159](https://github.com/pymc-devs/pymc3/pull/4159), [#4217](https://github.com/pymc-devs/pymc3/pull/4217), [#4243](https://github.com/pymc-devs/pymc3/pull/4243), and [#4260](https://github.com/pymc-devs/pymc3/pull/4260)).
- Add Discrete HyperGeometric Distribution (see [#4249](https://github.com/pymc-devs/pymc3/pull/#4249))

### Maintenance
- Switch the dependency of Theano to our own fork, [Theano-PyMC](https://github.com/pymc-devs/Theano-PyMC).
- Removed non-NDArray (Text, SQLite, HDF5) backends and associated tests.
- Use dill to serialize user defined logp functions in `DensityDist`. The previous serialization code fails if it is used in notebooks on Windows and Mac. `dill` is now a required dependency. (see [#3844](https://github.com/pymc-devs/pymc3/issues/3844)).
- Fixed numerical instability in ExGaussian's logp by preventing `logpow` from returning `-inf` (see [#4050](https://github.com/pymc-devs/pymc3/pull/4050)).
- Numerically improved stickbreaking transformation - e.g. for the `Dirichlet` distribution. [#4129](https://github.com/pymc-devs/pymc3/pull/4129)
- Enabled the `Multinomial` distribution to handle batch sizes that have more than 2 dimensions. [#4169](https://github.com/pymc-devs/pymc3/pull/4169)
- Test model logp before starting any MCMC chains (see [#4211](https://github.com/pymc-devs/pymc3/pull/4211))
- Fix bug in `model.check_test_point` that caused the `test_point` argument to be ignored. (see [PR #4211](https://github.com/pymc-devs/pymc3/pull/4211#issuecomment-727142721))
- Refactored MvNormal.random method with better handling of sample, batch and event shapes. [#4207](https://github.com/pymc-devs/pymc3/pull/4207)
- The `InverseGamma` distribution now implements a `logcdf`. [#3944](https://github.com/pymc-devs/pymc3/pull/3944)
- Make starting jitter methods for nuts sampling more robust by resampling values that lead to non-finite probabilities. A new optional argument `jitter-max-retries` can be passed to `pm.sample()` and `pm.init_nuts()` to control the maximum number of retries per chain. [4298](https://github.com/pymc-devs/pymc3/pull/4298)

### Documentation
- Added a new notebook demonstrating how to incorporate sampling from a conjugate Dirichlet-multinomial posterior density in conjunction with other step methods (see [#4199](https://github.com/pymc-devs/pymc3/pull/4199)).
- Mentioned the way to do any random walk with `theano.tensor.cumsum()` in `GaussianRandomWalk` docstrings (see [#4048](https://github.com/pymc-devs/pymc3/pull/4048)).

**Release manager** for 3.10.0: Eelke Spaak ([@Spaak](https://github.com/Spaak))

## PyMC3 3.9.3 (11 August 2020)

### New features
- Introduce optional arguments to `pm.sample`: `mp_ctx` to control how the processes for parallel sampling are started, and `pickle_backend` to specify which library is used to pickle models in parallel sampling when the multiprocessing context is not of type `fork` (see [#3991](https://github.com/pymc-devs/pymc3/pull/3991)).
- Add sampler stats `process_time_diff`, `perf_counter_diff` and `perf_counter_start`, that record wall and CPU times for each NUTS and HMC sample (see [ #3986](https://github.com/pymc-devs/pymc3/pull/3986)).
- Extend `keep_size` argument handling for `sample_posterior_predictive` and `fast_sample_posterior_predictive`, to work on ArviZ `InferenceData` and xarray `Dataset` input values (see [PR #4006](https://github.com/pymc-devs/pymc3/pull/4006) and issue [#4004](https://github.com/pymc-devs/pymc3/issues/4004)).
- SMC-ABC: add the Wasserstein and energy distance functions. Refactor API, the distance, sum_stats and epsilon arguments are now passed `pm.Simulator` instead of `pm.sample_smc`. Add random method to `pm.Simulator`. Add option to save the simulated data. Improved LaTeX representation [#3996](https://github.com/pymc-devs/pymc3/pull/3996).
- SMC-ABC: Allow use of potentials by adding them to the prior term. [#4016](https://github.com/pymc-devs/pymc3/pull/4016).

### Maintenance
- Fix an error on Windows and Mac where error message from unpickling models did not show up in the notebook, or where sampling froze when a worker process crashed (see [#3991](https://github.com/pymc-devs/pymc3/pull/3991)).
- Require Theano >= 1.0.5 (see [#4032](https://github.com/pymc-devs/pymc3/pull/4032)).

### Documentation
- Notebook on [multilevel modeling](https://docs.pymc.io/notebooks/multilevel_modeling.html) has been rewritten to showcase ArviZ and xarray usage for inference result analysis (see [#3963](https://github.com/pymc-devs/pymc3/pull/3963)).

_NB: The `docs/*` folder is still removed from the tarball due to an upload size limit on PyPi._

**Release manager** for 3.9.3: Kyle Beauchamp ([@kyleabeauchamp](https://github.com/kyleabeauchamp))

## PyMC3 3.9.2 (24 June 2020)

### Maintenance
- Warning added in GP module when `input_dim` is lower than the number of columns in `X` to compute the covariance function (see [#3974](https://github.com/pymc-devs/pymc3/pull/3974)).
- Pass the `tune` argument from `sample` when using `advi+adapt_diag_grad` (see issue [#3965](https://github.com/pymc-devs/pymc3/issues/3965), fixed by [#3979](https://github.com/pymc-devs/pymc3/pull/3979)).
- Add simple test case for new coords and dims feature in `pm.Model` (see [#3977](https://github.com/pymc-devs/pymc3/pull/3977)).
- Require ArviZ >= 0.9.0 (see [#3977](https://github.com/pymc-devs/pymc3/pull/3977)).
- Fixed issue [#3962](https://github.com/pymc-devs/pymc3/issues/3962) by making a change in the `_random()` method of `GaussianRandomWalk` class (see PR [#3985](https://github.com/pymc-devs/pymc3/pull/3985)). Further testing revealed a new issue which is being tracked by [#4010](https://github.com/pymc-devs/pymc3/issues/4010).

_NB: The `docs/*` folder is still removed from the tarball due to an upload size limit on PyPi._

**Release manager** for 3.9.2: Alex Andorra ([@AlexAndorra](https://github.com/AlexAndorra))

## PyMC3 3.9.1 (16 June 2020)
The `v3.9.0` upload to PyPI didn't include a tarball, which is fixed in this release.
Though we had to temporarily remove the `docs/*` folder from the tarball due to a size limit.

**Release manager** for 3.9.1: Michael Osthege ([@michaelosthege](https://github.com/michaelosthege))

## PyMC3 3.9.0 (16 June 2020)

### New features
- Use [fastprogress](https://github.com/fastai/fastprogress) instead of tqdm [#3693](https://github.com/pymc-devs/pymc3/pull/3693).
- `DEMetropolis` can now tune both `lambda` and `scaling` parameters, but by default neither of them are tuned. See [#3743](https://github.com/pymc-devs/pymc3/pull/3743) for more info.
- `DEMetropolisZ`, an improved variant of `DEMetropolis` brings better parallelization and higher efficiency with fewer chains with a slower initial convergence. This implementation is experimental. See [#3784](https://github.com/pymc-devs/pymc3/pull/3784) for more info.
- Notebooks that give insight into `DEMetropolis`, `DEMetropolisZ` and the `DifferentialEquation` interface are now located in the [Tutorials/Deep Dive](https://docs.pymc.io/nb_tutorials/index.html) section.
- Add `fast_sample_posterior_predictive`, a vectorized alternative to `sample_posterior_predictive`.  This alternative is substantially faster for large models.
- GP covariance functions can now be exponentiated by a scalar. See PR [#3852](https://github.com/pymc-devs/pymc3/pull/3852)
- `sample_posterior_predictive` can now feed on `xarray.Dataset` - e.g. from `InferenceData.posterior`. (see [#3846](https://github.com/pymc-devs/pymc3/pull/3846))
- `SamplerReport` (`MultiTrace.report`) now has properties `n_tune`, `n_draws`, `t_sampling` for increased convenience (see [#3827](https://github.com/pymc-devs/pymc3/pull/3827))
- `pm.sample(..., return_inferencedata=True)` can now directly return the trace as `arviz.InferenceData` (see [#3911](https://github.com/pymc-devs/pymc3/pull/3911))
- `pm.sample` now has support for adapting dense mass matrix using `QuadPotentialFullAdapt` (see [#3596](https://github.com/pymc-devs/pymc3/pull/3596), [#3705](https://github.com/pymc-devs/pymc3/pull/3705), [#3858](https://github.com/pymc-devs/pymc3/pull/3858), and [#3893](https://github.com/pymc-devs/pymc3/pull/3893)). Use `init="adapt_full"` or `init="jitter+adapt_full"` to use.
- `Moyal` distribution added (see [#3870](https://github.com/pymc-devs/pymc3/pull/3870)).
- `pm.LKJCholeskyCov` now automatically computes and returns the unpacked Cholesky decomposition, the correlations and the standard deviations of the covariance matrix (see [#3881](https://github.com/pymc-devs/pymc3/pull/3881)).
- `pm.Data` container can now be used for index variables, i.e with integer data and not only floats (issue [#3813](https://github.com/pymc-devs/pymc3/issues/3813), fixed by [#3925](https://github.com/pymc-devs/pymc3/pull/3925)).
- `pm.Data` container can now be used as input for other random variables (issue [#3842](https://github.com/pymc-devs/pymc3/issues/3842), fixed by [#3925](https://github.com/pymc-devs/pymc3/pull/3925)).
- Allow users to specify coordinates and dimension names instead of numerical shapes when specifying a model. This makes interoperability with ArviZ easier. ([see #3551](https://github.com/pymc-devs/pymc3/pull/3551))
- Plots and Stats API sections now link to ArviZ documentation [#3927](https://github.com/pymc-devs/pymc3/pull/3927)
- Add `SamplerReport` with properties `n_draws`, `t_sampling` and `n_tune` to SMC. `n_tune` is always 0 [#3931](https://github.com/pymc-devs/pymc3/issues/3931).
- SMC-ABC: add option to define summary statistics, allow to sample from more complex models, remove redundant distances [#3940](https://github.com/pymc-devs/pymc3/issues/3940)

### Maintenance
- Tuning results no longer leak into sequentially sampled `Metropolis` chains (see #3733 and #3796).
- We'll deprecate the `Text` and `SQLite` backends and the `save_trace`/`load_trace` functions, since this is now done with ArviZ. (see [#3902](https://github.com/pymc-devs/pymc3/pull/3902))
- ArviZ `v0.8.3` is now the minimum required version
- In named models, `pm.Data` objects now get model-relative names (see [#3843](https://github.com/pymc-devs/pymc3/pull/3843)).
- `pm.sample` now takes 1000 draws and 1000 tuning samples by default, instead of 500 previously (see [#3855](https://github.com/pymc-devs/pymc3/pull/3855)).
- Moved argument division out of `NegativeBinomial` `random` method. Fixes [#3864](https://github.com/pymc-devs/pymc3/issues/3864) in the style of [#3509](https://github.com/pymc-devs/pymc3/pull/3509).
- The Dirichlet distribution now raises a ValueError when it's initialized with <= 0 values (see [#3853](https://github.com/pymc-devs/pymc3/pull/3853)).
- Dtype bugfix in `MvNormal` and `MvStudentT` (see [3836](https://github.com/pymc-devs/pymc3/pull/3836)).
- End of sampling report now uses `arviz.InferenceData` internally and avoids storing
  pointwise log likelihood (see [#3883](https://github.com/pymc-devs/pymc3/pull/3883)).
- The multiprocessing start method on MacOS is now set to "forkserver", to avoid crashes (see issue [#3849](https://github.com/pymc-devs/pymc3/issues/3849), solved by [#3919](https://github.com/pymc-devs/pymc3/pull/3919)).
- The AR1 logp now uses the precision of the whole AR1 process instead of just the innovation precision (see issue [#3892](https://github.com/pymc-devs/pymc3/issues/3892), fixed by [#3899](https://github.com/pymc-devs/pymc3/pull/3899)).
- Forced the `Beta` distribution's `random` method to generate samples that are in the open interval $(0, 1)$, i.e. no value can be equal to zero or equal to one (issue [#3898](https://github.com/pymc-devs/pymc3/issues/3898) fixed by [#3924](https://github.com/pymc-devs/pymc3/pull/3924)).
- Fixed an issue that happened on Windows, that was introduced by the clipped beta distribution rvs function ([#3924](https://github.com/pymc-devs/pymc3/pull/3924)). Windows does not support the `float128` dtype, but we had assumed that it had to be available. The solution was to only support `float128` on Linux and Darwin systems (see issue [#3929](https://github.com/pymc-devs/pymc3/issues/3849) fixed by [#3930](https://github.com/pymc-devs/pymc3/pull/3930)).

### Deprecations
- Remove `sample_ppc` and `sample_ppc_w` that were deprecated in 3.6.
- Deprecated `sd` has been replaced by `sigma` (already in version 3.7) in continuous, mixed and timeseries distributions and now raises `DeprecationWarning` when `sd` is used. (see [#3837](https://github.com/pymc-devs/pymc3/pull/3837) and [#3688](https://github.com/pymc-devs/pymc3/issues/3688)).
- We'll deprecate the `Text` and `SQLite` backends and the `save_trace`/`load_trace` functions, since this is now done with ArviZ. (see [#3902](https://github.com/pymc-devs/pymc3/pull/3902))
- Dropped some deprecated kwargs and functions (see [#3906](https://github.com/pymc-devs/pymc3/pull/3906))
- Dropped the outdated 'nuts' initialization method for `pm.sample` (see [#3863](https://github.com/pymc-devs/pymc3/pull/3863)).

**Release manager** for 3.9.0: Michael Osthege ([@michaelosthege](https://github.com/michaelosthege))

## PyMC3 3.8 (November 29 2019)

### New features
- Implemented robust u turn check in NUTS (similar to stan-dev/stan#2800). See PR [#3605]
- Add capabilities to do inference on parameters in a differential equation with `DifferentialEquation`. See [#3590](https://github.com/pymc-devs/pymc3/pull/3590) and [#3634](https://github.com/pymc-devs/pymc3/pull/3634).
- Distinguish between `Data` and `Deterministic` variables when graphing models with graphviz. PR [#3491](https://github.com/pymc-devs/pymc3/pull/3491).
- Sequential Monte Carlo - Approximate Bayesian Computation step method is now available. The implementation is in an experimental stage and will be further improved.
- Added `Matern12` covariance function for Gaussian processes. This is the Matern kernel with nu=1/2.
- Progressbar reports number of divergences in real time, when available [#3547](https://github.com/pymc-devs/pymc3/pull/3547).
- Sampling from variational approximation now allows for alternative trace backends [#3550].
- Infix `@` operator now works with random variables and deterministics [#3619](https://github.com/pymc-devs/pymc3/pull/3619).
- [ArviZ](https://arviz-devs.github.io/arviz/) is now a requirement, and handles plotting, diagnostics, and statistical checks.
- Can use GaussianRandomWalk in sample_prior_predictive and sample_prior_predictive [#3682](https://github.com/pymc-devs/pymc3/pull/3682)
- Now 11 years of S&P returns in data set[#3682](https://github.com/pymc-devs/pymc3/pull/3682)

### Maintenance
- Moved math operations out of `Rice`, `TruncatedNormal`, `Triangular` and `ZeroInflatedNegativeBinomial` `random` methods. Math operations on values returned by `draw_values` might not broadcast well, and all the `size` aware broadcasting is left to `generate_samples`. Fixes [#3481](https://github.com/pymc-devs/pymc3/issues/3481) and [#3508](https://github.com/pymc-devs/pymc3/issues/3508)
- Parallelization of population steppers (`DEMetropolis`) is now set via the `cores` argument. ([#3559](https://github.com/pymc-devs/pymc3/pull/3559))
- Fixed a bug in `Categorical.logp`. In the case of multidimensional `p`'s, the indexing was done wrong leading to incorrectly shaped tensors that consumed `O(n**2)` memory instead of `O(n)`. This fixes issue [#3535](https://github.com/pymc-devs/pymc3/issues/3535)
- Fixed a defect in `OrderedLogistic.__init__` that unnecessarily increased the dimensionality of the underlying `p`. Related to issue issue [#3535](https://github.com/pymc-devs/pymc3/issues/3535) but was not the true cause of it.
- SMC: stabilize covariance matrix [3573](https://github.com/pymc-devs/pymc3/pull/3573)
- SMC: is no longer a step method of `pm.sample` now it should be called using `pm.sample_smc` [3579](https://github.com/pymc-devs/pymc3/pull/3579)
- SMC: improve computation of the proposal scaling factor [3594](https://github.com/pymc-devs/pymc3/pull/3594) and [3625](https://github.com/pymc-devs/pymc3/pull/3625)
- SMC: reduce number of logp evaluations [3600](https://github.com/pymc-devs/pymc3/pull/3600)
- SMC: remove `scaling` and `tune_scaling` arguments as is a better idea to always allow SMC to automatically compute the scaling factor [3625](https://github.com/pymc-devs/pymc3/pull/3625)
- Now uses `multiprocessong` rather than `psutil` to count CPUs, which results in reliable core counts on Chromebooks.
- `sample_posterior_predictive` now preallocates the memory required for its output to improve memory usage. Addresses problems raised in this [discourse thread](https://discourse.pymc.io/t/memory-error-with-posterior-predictive-sample/2891/4).
- Fixed a bug in `Categorical.logp`. In the case of multidimensional `p`'s, the indexing was done wrong leading to incorrectly shaped tensors that consumed `O(n**2)` memory instead of `O(n)`. This fixes issue [#3535](https://github.com/pymc-devs/pymc3/issues/3535)
- Fixed a defect in `OrderedLogistic.__init__` that unnecessarily increased the dimensionality of the underlying `p`. Related to issue issue [#3535](https://github.com/pymc-devs/pymc3/issues/3535) but was not the true cause of it.
- Wrapped `DensityDist.rand` with `generate_samples` to make it aware of the distribution's shape. Added control flow attributes to still be able to behave as in earlier versions, and to control how to interpret the `size` parameter in the `random` callable signature. Fixes [3553](https://github.com/pymc-devs/pymc3/issues/3553)
- Added `theano.gof.graph.Constant` to type checks done in `_draw_value` (fixes issue [3595](https://github.com/pymc-devs/pymc3/issues/3595))
- `HalfNormal` did not used to work properly in `draw_values`, `sample_prior_predictive`, or `sample_posterior_predictive` (fixes issue [3686](https://github.com/pymc-devs/pymc3/pull/3686))
- Random variable transforms were inadvertently left out of the API documentation. Added them. (See PR [3690](https://github.com/pymc-devs/pymc3/pull/3690)).
- Refactored `pymc3.model.get_named_nodes_and_relations` to use the ancestors and descendents, in a way that is consistent with `theano`'s naming convention.
- Changed the way in which `pymc3.model.get_named_nodes_and_relations` computes nodes without ancestors to make it robust to changes in var_name orderings (issue [#3643](https://github.com/pymc-devs/pymc3/issues/3643))

## PyMC3 3.7 (May 29 2019)

### New features

- Add data container class (`Data`) that wraps the theano SharedVariable class and let the model be aware of its inputs and outputs.
- Add function `set_data` to update variables defined as `Data`.
- `Mixture` now supports mixtures of multidimensional probability distributions, not just lists of 1D distributions.
- `GLM.from_formula` and `LinearComponent.from_formula` can extract variables from the calling scope. Customizable via the new `eval_env` argument. Fixing [#3382](https://github.com/pymc-devs/pymc3/issues/3382).
- Added the `distributions.shape_utils` module with functions used to help broadcast samples drawn from distributions using the `size` keyword argument.
- Used `numpy.vectorize` in `distributions.distribution._compile_theano_function`. This enables `sample_prior_predictive` and `sample_posterior_predictive` to ask for tuples of samples instead of just integers. This fixes issue [#3422](https://github.com/pymc-devs/pymc3/issues/3422).

### Maintenance

- All occurances of `sd` as a parameter name have been renamed to `sigma`. `sd` will continue to function for backwards compatibility.
- `HamiltonianMC` was ignoring certain arguments like `target_accept`, and not using the custom step size jitter function with expectation 1.
- Made `BrokenPipeError` for parallel sampling more verbose on Windows.
- Added the `broadcast_distribution_samples` function that helps broadcasting arrays of drawn samples, taking into account the requested `size` and the inferred distribution shape. This sometimes is needed by distributions that call several `rvs` separately within their `random` method, such as the `ZeroInflatedPoisson` (fixes issue [#3310](https://github.com/pymc-devs/pymc3/issues/3310)).
- The `Wald`, `Kumaraswamy`, `LogNormal`, `Pareto`, `Cauchy`, `HalfCauchy`, `Weibull` and `ExGaussian` distributions `random` method used a hidden `_random` function that was written with scalars in mind. This could potentially lead to artificial correlations between random draws. Added shape guards and broadcasting of the distribution samples to prevent this (Similar to issue [#3310](https://github.com/pymc-devs/pymc3/issues/3310)).
- Added a fix to allow the imputation of single missing values of observed data, which previously would fail (fixes issue [#3122](https://github.com/pymc-devs/pymc3/issues/3122)).
- The `draw_values` function was too permissive with what could be grabbed from inside `point`, which lead to an error when sampling posterior predictives of variables that depended on shared variables that had changed their shape after `pm.sample()` had been called (fix issue [#3346](https://github.com/pymc-devs/pymc3/issues/3346)).
- `draw_values` now adds the theano graph descendants of `TensorConstant` or `SharedVariables` to the named relationship nodes stack, only if these descendants are `ObservedRV` or `MultiObservedRV` instances (fixes issue [#3354](https://github.com/pymc-devs/pymc3/issues/3354)).
- Fixed bug in broadcast_distrution_samples, which did not handle correctly cases in which some samples did not have the size tuple prepended.
- Changed `MvNormal.random`'s usage of `tensordot` for Cholesky encoded covariances. This lead to wrong axis broadcasting and seemed to be the cause for issue [#3343](https://github.com/pymc-devs/pymc3/issues/3343).
- Fixed defect in `Mixture.random` when multidimensional mixtures were involved. The mixture component was not preserved across all the elements of the dimensions of the mixture. This meant that the correlations across elements within a given draw of the mixture were partly broken.
- Restructured `Mixture.random` to allow better use of vectorized calls to `comp_dists.random`.
- Added tests for mixtures of multidimensional distributions to the test suite.
- Fixed incorrect usage of `broadcast_distribution_samples` in `DiscreteWeibull`.
- `Mixture`'s default dtype is now determined by `theano.config.floatX`.
- `dist_math.random_choice` now handles nd-arrays of category probabilities, and also handles sizes that are not `None`. Also removed unused `k` kwarg from `dist_math.random_choice`.
- Changed `Categorical.mode` to preserve all the dimensions of `p` except the last one, which encodes each category's probability.
- Changed initialization of `Categorical.p`. `p` is now normalized to sum to `1` inside `logp` and `random`, but not during initialization. This could hide negative values supplied to `p` as mentioned in [#2082](https://github.com/pymc-devs/pymc3/issues/2082).
- `Categorical` now accepts elements of `p` equal to `0`. `logp` will return `-inf` if there are `values` that index to the zero probability categories.
- Add `sigma`, `tau`, and `sd` to signature of `NormalMixture`.
- Set default lower and upper values of -inf and inf for pm.distributions.continuous.TruncatedNormal. This avoids errors caused by their previous values of None (fixes issue [#3248](https://github.com/pymc-devs/pymc3/issues/3248)).
- Converted all calls to `pm.distributions.bound._ContinuousBounded` and `pm.distributions.bound._DiscreteBounded` to use only and all positional arguments (fixes issue [#3399](https://github.com/pymc-devs/pymc3/issues/3399)).
- Restructured `distributions.distribution.generate_samples` to use the `shape_utils` module. This solves issues [#3421](https://github.com/pymc-devs/pymc3/issues/3421) and [#3147](https://github.com/pymc-devs/pymc3/issues/3147) by using the `size` aware broadcating functions in `shape_utils`.
- Fixed the `Multinomial.random` and `Multinomial.random_` methods to make them compatible with the new `generate_samples` function. In the process, a bug of the `Multinomial.random_` shape handling was discovered and fixed.
- Fixed a defect found in `Bound.random` where the `point` dictionary was passed to `generate_samples` as an `arg` instead of in `not_broadcast_kwargs`.
- Fixed a defect found in `Bound.random_` where `total_size` could end up as a `float64` instead of being an integer if given `size=tuple()`.
- Fixed an issue in `model_graph` that caused construction of the graph of the model for rendering to hang: replaced a search over the powerset of the nodes with a breadth-first search over the nodes. Fix for [#3458](https://github.com/pymc-devs/pymc3/issues/3458).
- Removed variable annotations from `model_graph` but left type hints (Fix for [#3465](https://github.com/pymc-devs/pymc3/issues/3465)). This means that we support `python>=3.5.4`.
- Default `target_accept`for `HamiltonianMC` is now 0.65, as suggested in Beskos et. al. 2010 and Neal 2001.
- Fixed bug in `draw_values` that lead to intermittent errors in python3.5. This happened with some deterministic nodes that were drawn but not added to `givens`.

### Deprecations

- `nuts_kwargs` and `step_kwargs` have been deprecated in favor of using the standard `kwargs` to pass optional step method arguments.
- `SGFS` and `CSG` have been removed (Fix for [#3353](https://github.com/pymc-devs/pymc3/issues/3353)). They have been moved to [pymc3-experimental](https://github.com/pymc-devs/pymc3-experimental).
- References to `live_plot` and corresponding notebooks have been removed.
- Function `approx_hessian` was removed, due to `numdifftools` becoming incompatible with current `scipy`. The function was already optional, only available to a user who installed `numdifftools` separately, and not hit on any common codepaths. [#3485](https://github.com/pymc-devs/pymc3/pull/3485).
- Deprecated `vars` parameter of `sample_posterior_predictive` in favor of `varnames`.
-  References to `live_plot` and corresponding notebooks have been removed.
- Deprecated `vars` parameters of `sample_posterior_predictive` and `sample_prior_predictive` in favor of `var_names`.  At least for the latter, this is more accurate, since the `vars` parameter actually took names.

### Contributors sorted by number of commits
    45  Luciano Paz
    38  Thomas Wiecki
    23  Colin Carroll
    19  Junpeng Lao
    15  Chris Fonnesbeck
    13  Juan Martín Loyola
    13  Ravin Kumar
     8  Robert P. Goldman
     5  Tim Blazina
     4  chang111
     4  adamboche
     3  Eric Ma
     3  Osvaldo Martin
     3  Sanmitra Ghosh
     3  Saurav Shekhar
     3  chartl
     3  fredcallaway
     3  Demetri
     2  Daisuke Kondo
     2  David Brochart
     2  George Ho
     2  Vaibhav Sinha
     1  rpgoldman
     1  Adel Tomilova
     1  Adriaan van der Graaf
     1  Bas Nijholt
     1  Benjamin Wild
     1  Brigitta Sipocz
     1  Daniel Emaasit
     1  Hari
     1  Jeroen
     1  Joseph Willard
     1  Juan Martin Loyola
     1  Katrin Leinweber
     1  Lisa Martin
     1  M. Domenzain
     1  Matt Pitkin
     1  Peadar Coyle
     1  Rupal Sharma
     1  Tom Gilliss
     1  changjiangeng
     1  michaelosthege
     1  monsta
     1  579397

## PyMC3 3.6 (Dec 21 2018)

This will be the last release to support Python 2.

### New features

- Track the model log-likelihood as a sampler stat for NUTS and HMC samplers
  (accessible as `trace.get_sampler_stats('model_logp')`) (#3134)
- Add Incomplete Beta function `incomplete_beta(a, b, value)`
- Add log CDF functions to continuous distributions: `Beta`, `Cauchy`, `ExGaussian`, `Exponential`, `Flat`, `Gumbel`, `HalfCauchy`, `HalfFlat`, `HalfNormal`, `Laplace`, `Logistic`, `Lognormal`, `Normal`, `Pareto`, `StudentT`, `Triangular`, `Uniform`, `Wald`, `Weibull`.
- Behavior of `sample_posterior_predictive` is now to produce posterior predictive samples, in order, from all values of the `trace`. Previously, by default it would produce 1 chain worth of samples, using a random selection from the `trace` (#3212)
- Show diagnostics for initial energy errors in HMC and NUTS.
- PR #3273 has added the `distributions.distribution._DrawValuesContext` context
  manager. This is used to store the values already drawn in nested `random`
  and `draw_values` calls, enabling `draw_values` to draw samples from the
  joint probability distribution of RVs and not the marginals. Custom
  distributions that must call `draw_values` several times in their `random`
  method, or that invoke many calls to other distribution's `random` methods
  (e.g. mixtures) must do all of these calls under the same `_DrawValuesContext`
  context manager instance. If they do not, the conditional relations between
  the distribution's parameters could be broken, and `random` could return
  values drawn from an incorrect distribution.
- `Rice` distribution is now defined with either the noncentrality parameter or the shape parameter (#3287).

### Maintenance

- Big rewrite of documentation (#3275)
- Fixed Triangular distribution `c` attribute handling in `random` and updated sample codes for consistency (#3225)
- Refactor SMC and properly compute marginal likelihood (#3124)
- Removed use of deprecated `ymin` keyword in matplotlib's `Axes.set_ylim` (#3279)
- Fix for #3210. Now `distribution.draw_values(params)`, will draw the `params` values from their joint probability distribution and not from combinations of their marginals (Refer to PR #3273).
- Removed dependence on pandas-datareader for retrieving Yahoo Finance data in examples (#3262)
- Rewrote `Multinomial._random` method to better handle shape broadcasting (#3271)
- Fixed `Rice` distribution, which inconsistently mixed two parametrizations (#3286).
- `Rice` distribution now accepts multiple parameters and observations and is usable with NUTS (#3289).
- `sample_posterior_predictive` no longer calls `draw_values` to initialize the shape of the ppc trace. This called could lead to `ValueError`'s when sampling the ppc from a model with `Flat` or `HalfFlat` prior distributions (Fix issue #3294).
- Added explicit conversion to `floatX` and `int32` for the continuous and discrete probability distribution parameters (addresses issue #3223).


### Deprecations

- Renamed `sample_ppc()` and `sample_ppc_w()` to `sample_posterior_predictive()` and `sample_posterior_predictive_w()`, respectively.

## PyMC 3.5 (July 21 2018)

### New features

- Add documentation section on survival analysis and censored data models
- Add `check_test_point` method to `pm.Model`
- Add `Ordered` Transformation and `OrderedLogistic` distribution
- Add `Chain` transformation
- Improve error message `Mass matrix contains zeros on the diagonal. Some derivatives might always be zero` during tuning of `pm.sample`
- Improve error message `NaN occurred in optimization.` during ADVI
- Save and load traces without `pickle` using `pm.save_trace` and `pm.load_trace`
- Add `Kumaraswamy` distribution
- Add `TruncatedNormal` distribution
- Rewrite parallel sampling of multiple chains on py3. This resolves long standing issues when transferring large traces to the main process, avoids pickling issues on UNIX, and allows us to show a progress bar for all chains. If parallel sampling is interrupted, we now return partial results.
- Add `sample_prior_predictive` which allows for efficient sampling from the unconditioned model.
- SMC: remove experimental warning, allow sampling using `sample`, reduce autocorrelation from final trace.
- Add `model_to_graphviz` (which uses the optional dependency `graphviz`) to plot a directed graph of a PyMC3 model using plate notation.
- Add beta-ELBO variational inference as in beta-VAE model (Christopher P. Burgess et al. NIPS, 2017)
- Add `__dir__` to `SingleGroupApproximation` to improve autocompletion in interactive environments

### Fixes

- Fixed grammar in divergence warning, previously `There were 1 divergences ...` could be raised.
- Fixed `KeyError` raised when only subset of variables are specified to be recorded in the trace.
- Removed unused `repeat=None` arguments from all `random()` methods in distributions.
- Deprecated the `sigma` argument in `MarginalSparse.marginal_likelihood` in favor of `noise`
- Fixed unexpected behavior in `random`. Now the `random` functionality is more robust and will work better for `sample_prior` when that is implemented.
- Fixed `scale_cost_to_minibatch` behaviour, previously this was not working and always `False`

## PyMC 3.4.1 (April 18 2018)

### New features

- Add `logit_p` keyword to `pm.Bernoulli`, so that users can specify the logit of the success probability. This is faster and more stable than using `p=tt.nnet.sigmoid(logit_p)`.
- Add `random` keyword to `pm.DensityDist` thus enabling users to pass custom random method which in turn makes sampling from a `DensityDist` possible.
- Effective sample size computation is updated. The estimation uses Geyer's initial positive sequence, which no longer truncates the autocorrelation series inaccurately. `pm.diagnostics.effective_n` now can reports N_eff>N.
- Added `KroneckerNormal` distribution and a corresponding `MarginalKron` Gaussian Process implementation for efficient inference, along with lower-level functions such as `cartesian` and `kronecker` products.
- Added `Coregion` covariance function.
- Add new 'pairplot' function, for plotting scatter or hexbin matrices of sampled parameters. Optionally it can plot divergences.
- Plots of discrete distributions in the docstrings
- Add logitnormal distribution
- Densityplot: add support for discrete variables
- Fix the Binomial likelihood in `.glm.families.Binomial`, with the flexibility of specifying the `n`.
- Add `offset` kwarg to `.glm`.
- Changed the `compare` function to accept a dictionary of model-trace pairs instead of two separate lists of models and traces.
- add test and support for creating multivariate mixture and mixture of mixtures
- `distribution.draw_values`, now is also able to draw values from conditionally dependent RVs, such as autotransformed RVs (Refer to PR #2902).

### Fixes

- `VonMises` does not overflow for large values of kappa. i0 and i1 have been removed and we now use log_i0 to compute the logp.
- The bandwidth for KDE plots is computed using a modified version of Scott's rule. The new version uses entropy instead of standard deviation. This works better for multimodal distributions. Functions using KDE plots has a new argument `bw` controlling the bandwidth.
- fix PyMC3 variable is not replaced if provided in more_replacements (#2890)
- Fix for issue #2900. For many situations, named node-inputs do not have a `random` method, while some intermediate node may have it. This meant that if the named node-input at the leaf of the graph did not have a fixed value, `theano` would try to compile it and fail to find inputs, raising a `theano.gof.fg.MissingInputError`. This was fixed by going through the theano variable's owner inputs graph, trying to get intermediate named-nodes values if the leafs had failed.
- In `distribution.draw_values`, some named nodes could be `theano.tensor.TensorConstant`s or `theano.tensor.sharedvar.SharedVariable`s. Nevertheless, in `distribution._draw_value`, these would be passed to `distribution._compile_theano_function` as if they were `theano.tensor.TensorVariable`s. This could lead to the following exceptions `TypeError: ('Constants not allowed in param list', ...)` or `TypeError: Cannot use a shared variable (...)`. The fix was to not add `theano.tensor.TensorConstant` or `theano.tensor.sharedvar.SharedVariable` named nodes into the `givens` dict that could be used in `distribution._compile_theano_function`.
- Exponential support changed to include zero values.

### Deprecations

- DIC and BPIC calculations have been removed
- df_summary have been removed, use summary instead
- `njobs` and `nchains` kwarg are deprecated in favor of `cores` and `chains` for `sample`
- `lag` kwarg in `pm.stats.autocorr` and `pm.stats.autocov` is deprecated.


## PyMC 3.3 (January 9, 2018)

### New features

- Improve NUTS initialization `advi+adapt_diag_grad` and add `jitter+adapt_diag_grad` (#2643)
- Added `MatrixNormal` class for representing vectors of multivariate normal variables
- Implemented `HalfStudentT` distribution
- New benchmark suite added (see http://pandas.pydata.org/speed/pymc3/)
- Generalized random seed types
- Update loo, new improved algorithm (#2730)
- New CSG (Constant Stochastic Gradient) approximate posterior sampling algorithm (#2544)
- Michael Osthege added support for population-samplers and implemented differential evolution metropolis (`DEMetropolis`).  For models with correlated dimensions that can not use gradient-based samplers, the `DEMetropolis` sampler can give higher effective sampling rates. (also see [PR#2735](https://github.com/pymc-devs/pymc3/pull/2735))
- Forestplot supports multiple traces (#2736)
- Add new plot, densityplot (#2741)
- DIC and BPIC calculations have been deprecated
- Refactor HMC and implemented new warning system (#2677, #2808)

### Fixes

- Fixed `compareplot` to use `loo` output.
- Improved `posteriorplot` to scale fonts
- `sample_ppc_w` now broadcasts
- `df_summary` function renamed to `summary`
- Add test for `model.logp_array` and `model.bijection` (#2724)
- Fixed `sample_ppc` and `sample_ppc_w` to iterate all chains(#2633, #2748)
- Add Bayesian R2 score (for GLMs) `stats.r2_score` (#2696) and test (#2729).
- SMC works with transformed variables (#2755)
- Speedup OPVI (#2759)
- Multiple minor fixes and improvements in the docs (#2775, #2786, #2787, #2789, #2790, #2794, #2799, #2809)

### Deprecations

- Old (`minibatch-`)`advi` is removed (#2781)


## PyMC3 3.2 (October 10, 2017)

### New features

This version includes two major contributions from our Google Summer of Code 2017 students:

* Maxim Kochurov extended and refactored the variational inference module. This primarily adds two important classes, representing operator variational inference (`OPVI`) objects and `Approximation` objects. These make it easier to extend existing `variational` classes, and to derive inference from `variational` optimizations, respectively. The `variational` module now also includes normalizing flows (`NFVI`).
* Bill Engels added an extensive new Gaussian processes (`gp`) module. Standard GPs can be specified using either `Latent` or `Marginal` classes, depending on the nature of the underlying function. A Student-T process `TP` has been added. In order to accomodate larger datasets, approximate marginal Gaussian processes (`MarginalSparse`) have been added.

Documentation has been improved as the result of the project's monthly "docathons".

An experimental stochastic gradient Fisher scoring (`SGFS`) sampling step method has been added.

The API for `find_MAP` was enhanced.

SMC now estimates the marginal likelihood.

Added `Logistic` and `HalfFlat` distributions to set of continuous distributions.

Bayesian fraction of missing information (`bfmi`) function added to `stats`.

Enhancements to `compareplot` added.

QuadPotential adaptation has been implemented.

Script added to build and deploy documentation.

MAP estimates now available for transformed and non-transformed variables.

The `Constant` variable class has been deprecated, and will be removed in 3.3.

DIC and BPIC calculations have been sped up.

Arrays are now accepted as arguments for the `Bound` class.

`random` method was added to the `Wishart` and `LKJCorr` distributions.

Progress bars have been added to LOO and WAIC calculations.

All example notebooks updated to reflect changes in API since 3.1.

Parts of the test suite have been refactored.

### Fixes

Fixed sampler stats error in NUTS for non-RAM backends

Matplotlib is  no longer a hard dependency, making it easier to use in settings where installing Matplotlib is problematic. PyMC will only complain if plotting is attempted.

Several bugs in the Gaussian process covariance were fixed.

All chains are now used to calculate WAIC and LOO.

AR(1) log-likelihood function has been fixed.

Slice sampler fixed to sample from 1D conditionals.

Several docstring fixes.

### Contributors

The following people contributed to this release (ordered by number of commits):

Maxim Kochurov <maxim.v.kochurov@gmail.com>
Bill Engels <w.j.engels@gmail.com>
Chris Fonnesbeck <chris.fonnesbeck@vanderbilt.edu>
Junpeng Lao <junpeng.lao@unifr.ch>
Adrian Seyboldt <adrian.seyboldt@gmail.com>
AustinRochford <arochford@monetate.com>
Osvaldo Martin <aloctavodia@gmail.com>
Colin Carroll <colcarroll@gmail.com>
Hannes Vasyura-Bathke <hannes.bathke@gmx.net>
Thomas Wiecki <thomas.wiecki@gmail.com>
michaelosthege <thecakedev@hotmail.com>
Marco De Nadai <me@marcodena.it>
Kyle Beauchamp <kyleabeauchamp@gmail.com>
Massimo <mcavallaro@users.noreply.github.com>
ctm22396 <ctm22396@gmail.com>
Max Horn <maexlich@gmail.com>
Hennadii Madan <madanh2014@gmail.com>
Hassan Naseri <h.nasseri@gmail.com>
Peadar Coyle <peadarcoyle@googlemail.com>
Saurav R. Tuladhar <saurav@fastmail.com>
Shashank Shekhar <shashank.f1@gmail.com>
Eric Ma <ericmjl@users.noreply.github.com>
Ed Herbst <ed.herbst@gmail.com>
tsdlovell <dlovell@twosigma.com>
zaxtax <zaxtax@users.noreply.github.com>
Dan Nichol <daniel.nichol@univ.ox.ac.uk>
Benjamin Yetton <bdyetton@gmail.com>
jackhansom <jack.hansom@outlook.com>
Jack Tsai <jacksctsai@gmail.com>
Andrés Asensio Ramos <aasensioramos@gmail.com>


## PyMC3 3.1 (June 23, 2017)

### New features

* New user forum at http://discourse.pymc.io

* [Gaussian Process submodule](http://pymc-devs.github.io/pymc3/notebooks/GP-introduction.html)

* Much improved variational inference support:

  - [Add Operator Variational Inference (experimental).](http://pymc-devs.github.io/pymc3/notebooks/bayesian_neural_network_opvi-advi.html)

  - [Add Stein-Variational Gradient Descent as well as Amortized SVGD (experimental).](https://github.com/pymc-devs/pymc3/pull/2183)

  - [Add pm.Minibatch() to easily specify mini-batches.](http://pymc-devs.github.io/pymc3/notebooks/bayesian_neural_network_opvi-advi.html#Minibatch-ADVI)

  - Added various optimizers including ADAM.

  - Stopping criterion implemented via callbacks.

* sample() defaults changed: tuning is enabled for the first 500 samples which are then discarded from the trace as burn-in.

* MvNormal supports Cholesky Decomposition now for increased speed and numerical stability.

* Many optimizations and speed-ups.

* NUTS implementation now matches current Stan implementation.

* Add higher-order integrators for HMC.

* [Add sampler statistics.](http://pymc-devs.github.io/pymc3/notebooks/sampler-stats.html)

* [Add live-trace to see samples in real-time.](http://pymc-devs.github.io/pymc3/notebooks/live_sample_plots.html)

* ADVI stopping criterion implemented.

* Improved support for theano's floatX setting to enable GPU computations (work in progress).

* MvNormal supports Cholesky Decomposition now for increased speed and numerical stability.

* [Add Elliptical Slice Sampler.](http://pymc-devs.github.io/pymc3/notebooks/GP-slice-sampling.html)

* Added support for multidimensional minibatches

* [Sampled posteriors can now be turned into priors for Bayesian updating with a new interpolated distribution.](https://github.com/pymc-devs/pymc3/pull/2163)

* Added `Approximation` class and the ability to convert a sampled trace into an approximation via its `Empirical` subclass.

* `Model` can now be inherited from and act as a base class for user specified models (see pymc3.models.linear).

* Add MvGaussianRandomWalk and MvStudentTRandomWalk distributions.

* GLM models do not need a left-hand variable anymore.

* Refactored HMC and NUTS for better readability.

* Add support for Python 3.6.

### Fixes

* Bound now works for discrete distributions as well.

* Random sampling now returns the correct shape even for higher dimensional RVs.

* Use theano Psi and GammaLn functions to enable GPU support for them.


## PyMC3 3.0 (January 9, 2017)

We are proud and excited to release the first stable version of PyMC3, the product of more than [5 years](https://github.com/pymc-devs/pymc3/commit/85c7e06b6771c0d99cbc09cb68885cda8f7785cb) of ongoing development and contributions from over 80 individuals. PyMC3 is a Python module for Bayesian modeling which focuses on modern Bayesian computational methods, primarily gradient-based (Hamiltonian) MCMC sampling and variational inference. Models are specified in Python, which allows for great flexibility. The main technological difference in PyMC3 relative to previous versions is the reliance on Theano for the computational backend, rather than on Fortran extensions.

### New features

Since the beta release last year, the following improvements have been implemented:

* Added `variational` submodule, which features the automatic differentiation variational inference (ADVI) fitting method. Also supports mini-batch ADVI for large data sets. Much of this work was due to the efforts of Taku Yoshioka, and important guidance was provided by the Stan team (specifically Alp Kucukelbir and Daniel Lee).

* Added model checking utility functions, including leave-one-out (LOO) cross-validation, BPIC, WAIC, and DIC.

* Implemented posterior predictive sampling (`sample_ppc`).

* Implemented auto-assignment of step methods by `sample` function.

* Enhanced IPython Notebook examples, featuring more complete narratives accompanying code.

* Extensive debugging of NUTS sampler.

* Updated documentation to reflect changes in code since beta.

* Refactored test suite for better efficiency.

* Added von Mises, zero-inflated negative binomial, and Lewandowski, Kurowicka and Joe (LKJ) distributions.

* Adopted `joblib` for managing parallel computation of chains.

* Added contributor guidelines, contributor code of conduct and governance document.

### Deprecations

* Argument order of tau and sd was switched for distributions of the normal family:
- `Normal()`
- `Lognormal()`
- `HalfNormal()`

Old: `Normal(name, mu, tau)`
New: `Normal(name, mu, sd)` (supplying keyword arguments is unaffected).

* `MvNormal` calling signature changed:
Old: `MvNormal(name, mu, tau)`
New: `MvNormal(name, mu, cov)` (supplying keyword arguments is unaffected).

We on the PyMC3 core team would like to thank everyone for contributing and now feel that this is ready for the big time. We look forward to hearing about all the cool stuff you use PyMC3 for, and look forward to continued development on the package.

### Contributors

The following authors contributed to this release:

Chris Fonnesbeck <chris.fonnesbeck@vanderbilt.edu>
John Salvatier <jsalvatier@gmail.com>
Thomas Wiecki <thomas.wiecki@gmail.com>
Colin Carroll <colcarroll@gmail.com>
Maxim Kochurov <maxim.v.kochurov@gmail.com>
Taku Yoshioka <taku.yoshioka.4096@gmail.com>
Peadar Coyle (springcoil) <peadarcoyle@googlemail.com>
Austin Rochford <arochford@monetate.com>
Osvaldo Martin <aloctavodia@gmail.com>
Shashank Shekhar <shashank.f1@gmail.com>

In addition, the following community members contributed to this release:

A Kuz <for.akuz@gmail.com>
A. Flaxman <abie@alum.mit.edu>
Abraham Flaxman <abie@alum.mit.edu>
Alexey Goldin <alexey.goldin@gmail.com>
Anand Patil <anand.prabhakar.patil@gmail.com>
Andrea Zonca <code@andreazonca.com>
Andreas Klostermann <andreasklostermann@googlemail.com>
Andres Asensio Ramos
Andrew Clegg <andrew.clegg@pearson.com>
Anjum48
Benjamin Edwards <bedwards@cs.unm.edu>
Boris Avdeev <borisaqua@gmail.com>
Brian Naughton <briannaughton@gmail.com>
Byron Smith
Chad Heyne <chadheyne@gmail.com>
Corey Farwell <coreyf@rwell.org>
David Huard <david.huard@gmail.com>
David Stück <dstuck@users.noreply.github.com>
DeliciousHair <mshepit@gmail.com>
Dustin Tran
Eigenblutwurst <Hannes.Bathke@gmx.net>
Gideon Wulfsohn <gideon.wulfsohn@gmail.com>
Gil Raphaelli <g@raphaelli.com>
Gogs <gogitservice@gmail.com>
Ilan Man
Imri Sofer <imrisofer@gmail.com>
Jake Biesinger <jake.biesinger@gmail.com>
James Webber <jamestwebber@gmail.com>
John McDonnell <john.v.mcdonnell@gmail.com>
Jon Sedar <jon.sedar@applied.ai>
Jordi Diaz
Jordi Warmenhoven <jordi.warmenhoven@gmail.com>
Karlson Pfannschmidt <kiudee@mail.uni-paderborn.de>
Kyle Bishop <citizenphnix@gmail.com>
Kyle Meyer <kyle@kyleam.com>
Lin Xiao
Mack Sweeney <mackenzie.sweeney@gmail.com>
Matthew Emmett <memmett@unc.edu>
Michael Gallaspy <gallaspy.michael@gmail.com>
Nick <nalourie@example.com>
Osvaldo Martin <aloctavodia@gmail.com>
Patricio Benavente <patbenavente@gmail.com>
Raymond Roberts
Rodrigo Benenson <rodrigo.benenson@gmail.com>
Sergei Lebedev <superbobry@gmail.com>
Skipper Seabold <chris.fonnesbeck@vanderbilt.edu>
Thomas Kluyver <takowl@gmail.com>
Tobias Knuth <mail@tobiasknuth.de>
Volodymyr Kazantsev
Wes McKinney <wesmckinn@gmail.com>
Zach Ploskey <zploskey@gmail.com>
akuz <for.akuz@gmail.com>
brandon willard <brandonwillard@gmail.com>
dstuck <dstuck88@gmail.com>
ingmarschuster <ingmar.schuster.linguistics@gmail.com>
jan-matthis <mail@jan-matthis.de>
jason <JasonTam22@gmailcom>
kiudee <quietdeath@gmail.com>
maahnman <github@mm.maahn.de>
macgyver <neil.rabinowitz@merton.ox.ac.uk>
mwibrow <mwibrow@gmail.com>
olafSmits <o.smits@gmail.com>
paul sorenson <paul@metrak.com>
redst4r <redst4r@web.de>
santon <steven.anton@idanalytics.com>
sgenoud <stevegenoud+github@gmail.com>
stonebig <stonebig>
Tal Yarkoni <tyarkoni@gmail.com>
x2apps <x2apps@yahoo.com>
zenourn <daniel@zeno.co.nz>

## PyMC3 3.0b (June 16th, 2015)

Probabilistic programming allows for flexible specification of Bayesian statistical models in code. PyMC3 is a new, open-source probabilistic programmer framework with an intuitive, readable and concise, yet powerful, syntax that is close to the natural notation statisticians use to describe models. It features next-generation fitting techniques, such as the No U-Turn Sampler, that allow fitting complex models with thousands of parameters without specialized knowledge of fitting algorithms.

PyMC3 has recently seen rapid development. With the addition of two new major features: automatic transforms and missing value imputation, PyMC3 has become ready for wider use. PyMC3 is now refined enough that adding features is easy, so we don't expect adding features in the future will require drastic changes. It has also become user friendly enough for a broader audience. Automatic transformations mean NUTS and find_MAP work with less effort, and friendly error messages mean its easy to diagnose problems with your model.

Thus, Thomas, Chris and I are pleased to announce that PyMC3 is now in Beta.

### Highlights
* Transforms now automatically applied to constrained distributions
* Transforms now specified with a `transform=` argument on Distributions. `model.TransformedVar` is gone.
* Transparent missing value imputation support added with MaskedArrays or pandas.DataFrame NaNs.
* Bad default values now ignored
* Profile theano functions using `model.profile(model.logpt)`

### Contributors since 3.0a
* A. Flaxman <abie@alum.mit.edu>
* Andrea Zonca <code@andreazonca.com>
* Andreas Klostermann <andreasklostermann@googlemail.com>
* Andrew Clegg <andrew.clegg@pearson.com>
* AustinRochford <arochford@monetate.com>
* Benjamin Edwards <bedwards@cs.unm.edu>
* Brian Naughton <briannaughton@gmail.com>
* Chad Heyne <chadheyne@gmail.com>
* Chris Fonnesbeck <chris.fonnesbeck@vanderbilt.edu>
* Corey Farwell <coreyf@rwell.org>
* John Salvatier <jsalvatier@gmail.com>
* Karlson Pfannschmidt <quietdeath@gmail.com>
* Kyle Bishop <citizenphnix@gmail.com>
* Kyle Meyer <kyle@kyleam.com>
* Mack Sweeney <mackenzie.sweeney@gmail.com>
* Osvaldo Martin <aloctavodia@gmail.com>
* Raymond Roberts <rayvroberts@gmail.com>
* Rodrigo Benenson <rodrigo.benenson@gmail.com>
* Thomas Wiecki <thomas.wiecki@gmail.com>
* Zach Ploskey <zploskey@gmail.com>
* maahnman <github@mm.maahn.de>
* paul sorenson <paul@metrak.com>
* zenourn <daniel@zeno.co.nz>
