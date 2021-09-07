# Glossary

A glossary of common terms used throughout the PyMC3 documentation and examples.

:::::{glossary}
[Term with external reference](https://www.youtube.com/watch?v=6dc7JgR8eI0)
  Terms are defined within this glossary directive. The term id is defined as the non
  indented line, and can be text alone (like {term}`second term`) or also include a link
  to an external reference.

Second term
  Definitions can have as many lines as desired, and should be written in markdown. Definitions
  can contain any markdown formatting for MyST to parse, this includes basic markdown like **bold**
  as well as MyST directives and roles like {fa}`fort awesome,style=fab`
Functional Programming
  Functional programming is a programming style that prefers the use of basic functions with explicit and distinct inputs and outputs.
  This contrasts with functions or methods that depend on variables that are not explicitly passed as an input (such as accessing `self.variable` inside a method) or that alter the inputs or other state variables in-place, instead of returning new distinct variables as outputs.
Dispatching
  Choosing which function or method implementation to use based on the type of the input variables (usually just the first variable). For some examples, see Python's documentation for the [singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch) decorator.
  
Bayesian model
  A Bayesian model is a composite of variables and distributional definitions for these variables. Fundamentally, models are mappings of one set of variables through a probability distribution onto another set of variables. These models define the ways values of some variables can arise, given values of other variables.
  
Bayesian data analysis
  Once we have defined the statistical model, Bayesian data analysis logically processes the data to produce inference. Probability theory is used to represent plausibility, whether in reference to countable events in the world, or theoretical constructs like parameters.
  
Likelihood
  The probability of the data, or the relative number of ways that a value of a parameter can produce the data, is called a likelihood. It is derived by enumerating all the possible data sequences that could have happened and then eliminating those sequences inconsistent with the data.
  
Prior
  Bayesian statistics allow us to include all information we have about the structure of the problem into the model. We do this via assuming prior distributions of the model’s variables, or the priors. Priors represent the plausibility of each possible value of the parameters before accounting for the data. Priors multiplied by likelihood produces the posterior. 
  
Posterior
  The outcome of a Bayesian model is the posterior probability, the updated plausibility of any specific target of inference after the model has seen the data. It produces the posterior distribution, which is a ranking of the relative plausibilities of every possible combination of parameter values.

  Once your model produces a posterior distribution, you need to summarize and interpret it. Common questions for the posterior include: intervals of defined boundaries, intervals of defined probability mass, point estimates. When the posterior distribution is very similar to the prior, the available data does not contain much information about a parameter of interest.

:::::
