# Glossary

A glossary of common terms used throughout the PyMC3 documentation and examples.

:::::{glossary}
[Equidispersion](http://www.ce.memphis.edu/7012/L20_CountDataModels_v2.pdf)
  If in a Poisson distribution if the variance equals the mean of the distribution, it is reffered to as equidispersion.

[Generalized Poisson PMF](https://www.jstor.org/stable/1267389)
  A generalization of the {term}`Poisson distribution`, with two parameters X1, and X2, is obtained as a limiting form of the {term}`generalized negative binomial distribution`. The variance of the distribution is greater than, equal to or smaller than the mean according as X2 is positive, zero or negative. For formula and more detail, visit the link in the title.

[Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)
  Describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately (by conditioning it on their age) than simply assuming that the individual is typical of the population as a whole.
  Formula:
  {term}`P(A|B) = (P(B|A) P(A))/P(B)`
  Where A and B are events and P(B) != 0
  

[Markov Chain](https://setosa.io/ev/markov-chains/)
  A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. For a visual explantation, visit the link in the title.

[Markov Chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
  Markov chain Monte Carlo (MCMC) methods comprise a class of algorithms for sampling from a probability distribution. By constructing a {term}`Markov Chain` that has the desired distribution as its equilibrium distribution, one can obtain a sample of the desired distribution by recording states from the chain.  Various algorithms exist for constructing chains, including the Metropolis–Hastings algorithm.
:::::