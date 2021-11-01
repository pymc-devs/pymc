# Glossary

A glossary of common terms used throughout the PyMC documentation and examples.

:::::{glossary}

Functional Programming
  Functional programming is a programming style that prefers the use of basic functions with explicit and distinct inputs and outputs.
  This contrasts with functions or methods that depend on variables that are not explicitly passed as an input (such as accessing `self.variable` inside a method) or that alter the inputs or other state variables in-place, instead of returning new distinct variables as outputs.
Dispatching
  Choosing which function or method implementation to use based on the type of the input variables (usually just the first variable). For some examples, see Python's documentation for the [singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch) decorator.

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

  - For more on generating and interpreting posterior samples, see Statistical Rethinking 2nd Edition By Richard McElreath, chapter 3.

Maximum a Posteriori
MAP
  A [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) is a point-estimate of an unknown quantity, that equals the mode of the posterior distribution. If the prior distribution is a uniform distribution, the MAP method is numerically equivalent to the Maximum Likelihood Estimate ({term}`MLE`). When the prior is not uniform the MAP estimation can be seen as a regularized version of the MLE.

-  - For a concise comparison between {term}`MLE` and {term}`MAP`, consider Deep Learning by Ian Goodfellow, chapter 5.6.1 or [Machine Learning: a Probabilistic Perspective](https://probml.github.io/pml-book/book1.html) by Kevin Murphy.

:::::
