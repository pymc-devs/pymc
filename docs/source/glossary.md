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
:::::

Equidispersion
  Equidispersion exists when data exibits variation similar to what you would expect based on a binomial distribution (for defectives) or a Poisson distribution (for defects). Traditional P charts and U charts assume that your rate of defectives or defects remains constant over time.

Generalized Poisson PMF
  A generalization of the Poisson distribution, with two parameters X1, and X2, is obtained as a limiting form of the generalized negative binomial distribution. The variance of the distribution is greater than, equal to or smaller than the mean according. as X2 is positive, zero or negative.

Bayes' theorem
  Describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately (by conditioning it on their age) than simply assuming that the individual is typical of the population as a whole.

Markov Chain (MC)
  A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event.

Markov Chain Monte Carlo (MCMC)
  Markov chain Monte Carlo (MCMC) methods comprise a class of algorithms for sampling from a probability distribution. By constructing a Markov chain that has the desired distribution as its equilibrium distribution, one can obtain a sample of the desired distribution by recording states from the chain.  Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm. Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm.