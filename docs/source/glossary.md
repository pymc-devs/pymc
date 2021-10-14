# Glossary

A glossary of common terms used throughout the PyMC documentation and examples.

:::::{glossary}
:sorted:

Functional Programming
  Functional programming is a programming style that prefers the use of basic functions with explicit and distinct inputs and outputs.
  This contrasts with functions or methods that depend on variables that are not explicitly passed as an input (such as accessing `self.variable` inside a method) or that alter the inputs or other state variables in-place, instead of returning new distinct variables as outputs.

Dispatching
  Choosing which function or method implementation to use based on the type of the input variables (usually just the first variable). For some examples, see Python's documentation for the [singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch) decorator.

[Dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion)
  In statistics, dispersion (also called variability, scatter, or spread) is the extent to which a distribution is stretched or squeezed

[Overdispersion](https://en.wikipedia.org/wiki/Overdispersion)
  In statistics, overdispersion is the presence of greater {term}`variability <dispersion>` in a data set than would be expected based on a given statistical model.

Underdispersion
  In statistics, underdispersion is the presence of lower {term}`variability <dispersion>` in a data set than would be expected based on a given statistical model.

Bayesian Workflow
  Bayesian workflow is the overall iterative procedure towards model refinement. It often includes the two related tasks of {term}`inference` and the exploratory analysis of models.
  - For a compact overview, see Bayesian statistics and modelling by van de Schoot, R., Depaoli, S., King, R. et al in Nat Rev Methods - Primers 1, 1 (2021).
  - For an in-depth overview, see Bayesian Workflow by Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, Martin Modrák
  - For an exercise-based material, see Think Bayes 2e: Bayesian Statistics Made Simple by Allen B. Downey
  - For an upcoming textbook that uses PyMC3, Tensorflow Probability, and ArviZ libraries, see Bayesian Modeling and Computation by Osvaldo A. Martin, Ravin Kumar, Junpeng Lao

Bayesian inference
  Once we have defined the statistical model, Bayesian inference processes the data and model to produce a {term}`posterior` distribution. That is a joint distribution of all parameters in the model. This distribution is used to represent plausibility, and is the logical consequence of the model and data.

Bayesian model
  A Bayesian model is a composite of variables and distributional definitions for these variables. Fundamentally, it tells you all the ways that the observed data could have been produced.

Prior
  Bayesian statistics allow us, in principle, to include all information we have about the structure of the problem into the model. We can do this via assuming prior distributions of the model’s parameters. Priors represent the plausibility of the value of the parameters before accounting for the data. Priors multiplied by {term}`likelihood` produce the {term}`posterior`.

  Priors’ informativeness can fall anywhere on the complete uncertainty to relative certainty continuum. An informative prior might encode known restrictions on the possible range of values of that parameter.

  To understand the implications of a prior, as well as the model itself, we can simulate predictions from the model, using only the prior distribution instead of the {term}`posterior` distribution - a process sometimes referred to as prior predictive simulation.

  - For an in-depth guide to priors, consider Statistical Rethinking 2nd Edition By Richard McElreath, especially chapters 2.3

Likelihood
  There are many perspectives on likelihood, but conceptually we can think about it as the relative number of ways the model could have produced the data; in other words, the probability of the data, given the parameters.

  - For an in-depth unfolding of the concept, refer to Statistical Rethinking 2nd Edition By Richard McElreath, particularly chapter 2.
  - For the problem-based material, see Think Bayes 2e: Bayesian Statistics Made Simple by Allen B. Downey
  - For univariate, continuous scenarios, see the calibr8 paper: Bayesian calibration, process modeling and uncertainty quantification in biotechnology by Laura Marie Helleckes,  Michael Osthege, Wolfgang Wiechert, Eric von Lieres, Marco Oldiges

Posterior
  The outcome of a Bayesian model is the posterior distribution, which describes the relative plausibilities of every possible combination of parameter values. We can think of the posterior as the updated {term}`priors` after the model has seen the data.

  When the posterior is obtained using numerical methods we first need to check how adequately the model fits to data. By sampling from the posterior distribution we can simulate the observations, or the implied predictions of the model. This posterior predictive distribution can then be compared to the observed data, the process known as the posterior predictive check.

  Once you are satisfied with the model, posterior distribution can be summarized and interpreted. Common questions for the posterior include: intervals of defined boundaries, intervals of defined probability mass, and point estimates. When the posterior is very similar to the prior, the available data does not contain much information about a parameter of interest.

  - For more on generating and interpreting the posterior samples, see Statistical Rethinking 2nd Edition By Richard McElreath, chapter 3.

Generalized Linear Model
GLM
  In a Generalized Linear Model (GLM), we assume the response variable $y_i$ to follow an
  exponential family distribution with mean $\mu_i$, which is assumed to be some (often nonlinear)
  function of $x_i^T\beta$. They're considered linear because the covariates affect the distribution
  of $Y_i$ only through the linear combination $x_i^T\beta$. Some examples of Generalized Linear
  Models are: Linear Regression, ANOVA, Logistic Regression and Poisson Regression

  :::{note} Do not confuse these with general linear models
  :::

[Probability Mass Function](https://en.wikipedia.org/wiki/Probability_mass_function)
[PMF](https://en.wikipedia.org/wiki/Probability_mass_function)
  A function that gives the probability that a discrete random variable is exactly equal to some value.

[Maximum a Posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)
[MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)
  An estimate of an unknown density estimation, that equals the mode of the posterior distribution. MAP can therefore be seen as a regularization of maximum likelihood estimation.

[No-U-Turn Sampler](https://arxiv.org/abs/1111.4246)
[NUTS](https://arxiv.org/abs/1111.4246)
  An extension of {term}`Hamiltonian Monte Carlo` that algorithmically sets likely candidate points that spans a wide swath of the target distribution, stopping automatically when it starts to double back and retrace its steps.

[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
[HMC](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
  A {term}`Markov Chain Monte Carlo` method for obtaining a sequence of random samples which converge to being distributed according to a target probability distribution.

[Credibility](https://en.wikipedia.org/wiki/Credibility_theory)
  A form of statistical inference used to forecast an uncertain future event

[Ordinary Differential Equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation)
[ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation)
  A type of differential equation containing one or more functions of one independent variable and the derivatives of those functions

Hierarchical Ordinary Differential Equation
  Individual, group, or other level types calculations of {term}`Ordinary Differential Equation`'s.
:::::
