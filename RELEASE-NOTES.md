# Release Notes

## PyMC 3.4 (unreleased)

### New features

- Add `logit_p` keyword to `pm.Bernoulli`, so that users can specify the logit of the success probability. This is faster and more stable than using `p=tt.nnet.sigmoid(logit_p)`.
- Add `random` keyword to `pm.DensityDist` thus enabling users to pass custom random method which in turn makes sampling from a `DensityDist` possible.
- Effective sample size computation is updated. The estimation uses Geyer's initial positive sequence, which no longer truncates the autocorrelation series inaccurately. `pm.diagnostics.effective_n` now can reports N_eff>N.
- Added `KroneckerNormal` distribution and a corresponding `MarginalKron`
  Gaussian Process implementation for efficient inference, along with
  lower-level functions such as `cartesian` and `kronecker` products.
- Added `Coregion` covariance function.
- Add new 'pairplot' function, for plotting scatter or hexbin matrices of sampled parameters.
  Optionally it can plot divergences.
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
