(glossary)=
# Glossary

A glossary of common terms used throughout the PyMC documentation and examples.

:::::{glossary}
:sorted:

Functional Programming
  Functional programming is a programming style that prefers the use of basic functions with explicit and distinct inputs and outputs.
  This contrasts with functions or methods that depend on variables that are not explicitly passed as an input (such as accessing `self.variable` inside a method) or that alter the inputs or other state variables in-place, instead of returning new distinct variables as outputs.

Dispatching
  Choosing which function or method implementation to use based on the type of the input variables (usually just the first variable). For some examples, see Python's documentation for the {func}`singledispatch <~functools.singledispatch>` decorator.

[Dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion)
  In statistics, dispersion (also called variability, scatter, or spread) is the extent to which a distribution is stretched or squeezed

[Overdispersion](https://en.wikipedia.org/wiki/Overdispersion)
  In statistics, overdispersion is the presence of greater {term}`variability <dispersion>` in a data set than would be expected based on a given statistical model.

Underdispersion
  In statistics, underdispersion is the presence of lower {term}`variability <dispersion>` in a data set than would be expected based on a given statistical model.

Bayesian Workflow
  The Bayesian workflow involves all the steps needed for model building. This includes {term}`Bayesian inference` but also other tasks such as i) diagnoses of the quality of the inference, ii) model criticism, including evaluations of both model assumptions and model predictions, iii) comparison of models, not
  just for the purpose of model selection or model averaging but more importantly to better understand these models and iv) Preparation of the results for a particular audience. These non-inferencial tasks require both numerical and visual summaries to help practitioners analyse their models. And they are sometimes collectively known as [Exploratory Analysis of Bayesian Models](https://joss.theoj.org/papers/10.21105/joss.01143).
  - For a compact overview, see Bayesian statistics and modelling by van de Schoot, R., Depaoli, S., King, R. et al in Nat Rev Methods - Primers 1, 1 (2021).
  - For an in-depth overview, see Bayesian Workflow by Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, Martin Modrák
  - For an exercise-based material, see Think Bayes 2e: Bayesian Statistics Made Simple by Allen B. Downey
  - For an upcoming textbook that uses PyMC, Tensorflow Probability, and ArviZ libraries, see Bayesian Modeling and Computation by Osvaldo A. Martin, Ravin Kumar and Junpeng Lao

Bayesian inference
  Once we have defined the statistical model, Bayesian inference processes the data and model to produce a {term}`posterior` distribution. That is a joint distribution of all parameters in the model. This distribution is used to represent plausibility, and is the logical consequence of the model and data.

Bayesian model
  A Bayesian model is a composite of variables and distributional definitions for these variables. Bayesian models have two defining characteristics: i) Unknown quantities are described using probability distributions and ii) Bayes' theorem is used to update the values of the parameters conditioned on the data

Prior
  Bayesian statistics allow us, in principle, to include all information we have about the structure of the problem into a model. We can do this via assuming prior distributions of the model’s parameters. Priors represent the plausibility of the value of the parameters before accounting for the data. Priors multiplied by {term}`likelihood` produce the {term}`posterior`.

  Priors’ informativeness can fall anywhere on the complete uncertainty to relative certainty continuum. An informative prior might encode known restrictions on the possible range of values of that parameter.

  To understand the implications of a prior and likelihood we can simulate predictions from the model, before seeing any data. This can be done by taking samples from the prior predictive distribution.

  - For an in-depth guide to priors, consider Statistical Rethinking 2nd Edition By Richard McElreath, especially chapters 2.3

Likelihood
  There are many perspectives on likelihood, but conceptually we can think about it as the probability of the data, given the parameters. Or in other words, as the relative number of ways the data could have been produced.

  - For an in-depth unfolding of the concept, refer to Statistical Rethinking 2nd Edition By Richard McElreath, particularly chapter 2.
  - For the problem-based material, see Think Bayes 2e: Bayesian Statistics Made Simple by Allen B. Downey
  - For univariate, continuous scenarios, see the calibr8 paper: Bayesian calibration, process modeling and uncertainty quantification in biotechnology by Laura Marie Helleckes,  Michael Osthege, Wolfgang Wiechert, Eric von Lieres, Marco Oldiges

Posterior
  The outcome of Bayesian inference is a posterior distribution, which describes the relative plausibilities of every possible combination of parameter values, given the observed data. We can think of the posterior as the updated {term}`priors` after the model has seen the data.

  When the posterior is obtained using numerical methods we generally need to first diagnose the quality of the computed approximation. This is necessary as, for example, methods like {term}`MCMC` has only asymptotic guarantees. In a Bayesian setting predictions can be simulated by sampling from the posterior predictive distribution. When such predictions are used to check the internal consistency of the models by comparing it with the observed data used for inference, the process is known as the posterior predictive checks.

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
  It is a point-estimate of an unknown quantity, that equals the mode of the posterior distribution.

  If the prior distribution is a flat distribution, the MAP method is numerically equivalent to the {term}`Maximum Likelihood Estimate` (MLE).
  When the prior is not flat the MAP estimation can be seen as a regularized version of the MLE.

  - For a concise comparison between {term}`MLE` and {term}`MAP`, consider Deep Learning by Ian Goodfellow, chapter 5.6.1 or [Machine Learning: a Probabilistic Perspective](https://probml.github.io/pml-book/book1.html) by Kevin Murphy.

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

[Generalized Poisson Distribution](https://doi.org/10.2307/1267389)
  A generalization of the {term}`Poisson distribution`, with two parameters X1, and X2, is obtained as a limiting form of the generalized negative binomial distribution. The variance of the distribution is greater than, equal to or smaller than the mean according as X2 is positive, zero or negative. For formula and more detail, visit the link in the title.

[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
  Describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately (by conditioning it on their age) than simply assuming that the individual is typical of the population as a whole.
  Formula:

  $$
  P(A|B) = \frac{P(B|A) P(A)}{P(B)}
  $$

  Where $A$ and $B$ are events and $P(B) \neq 0$


[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
  A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event.

[Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
[MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
  Markov chain Monte Carlo (MCMC) methods comprise a class of algorithms for sampling from a probability distribution. By constructing a {term}`Markov Chain` that has the desired distribution as its equilibrium distribution, one can obtain a sample of the desired distribution by recording states from the chain.  Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm.

tensor_like
  Any scalar or sequence that can be interpreted as a {class}`~pytensor.tensor.TensorVariable`. In addition to TensorVariables, this includes NumPy ndarrays, scalars, lists and tuples (possibly nested). Any argument accepted by `pytensor.tensor.as_tensor_variable` is tensor_like.

  ```{jupyter-execute}
  import pytensor.tensor as pt

  pt.as_tensor_variable([[1, 2.0], [0, 0]])
  ```

unnamed_distribution
    PyMC distributions can be initialized directly (e.g. `pm.Normal`) or using the `.dist` classmethod (e.g. `pm.Normal.dist`). Distributions initialized with the 1st method are registered as model parameters and thus, need to be given a name and be initialized within a model context. "unnamed_distributions" are distributions initialized with the 2nd method. These are standalone distributions, they are not parameters in any model and can be used to draw samples from a distribution by itself or as parameters to other distributions like mixtures or censored.

    "unnamed_distributions" can be used outside the model context. For example:

    ```{jupyter-execute}
    import pymc as pm

    unnamed_dist = pm.Normal.dist(mu=1, sigma=2)
    pm.draw(unnamed_dist, draws=10)
    ```

    Trying to initialize a named distribution outside a model context raises a `TypeError`:

    ```{jupyter-execute}
    :raises: TypeError

    import pymc as pm

    pm.Normal("variable")
    ```

:::::
