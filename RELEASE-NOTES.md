# Release Notes

## PyMC3 3.7 (unreleased)

### New features

### Maintenance

- All occurances of `sd` as a parameter name have been renamed to `sigma`. `sd` will continue to function for backwards compatibility.
- Made `BrokenPipeError` for parallel sampling more verbose on Windows.
- Added the `broadcast_distribution_samples` function that helps broadcasting arrays of drawn samples, taking into account the requested `size` and the inferred distribution shape. This sometimes is needed by distributions that call several `rvs` separately within their `random` method, such as the `ZeroInflatedPoisson` (Fix issue #3310).
- The `Wald`, `Kumaraswamy`, `LogNormal`, `Pareto`, `Cauchy`, `HalfCauchy`, `Weibull` and `ExGaussian` distributions `random` method used a hidden `_random` function that was written with scalars in mind. This could potentially lead to artificial correlations between random draws. Added shape guards and broadcasting of the distribution samples to prevent this (Similar to issue #3310).

### Deprecations

- `nuts_kwargs` and `step_kwargs` have been deprecated in favor of using the standard `kwargs` to pass optional step method arguments.

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

