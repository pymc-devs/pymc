**********************
Monte-Carlo Experiment
**********************

MCEx is an experimental package designed to be allow experimentation with how an MCMC package should be designed. It's goal is to be simple to use, understand, extend and improve, while still being fast. The hope is that some of the lessons learned in this experimental package lead to improvements in PyMC.

MCEx outsources its computational core to Theano. The advantage of this is that Theano is that MCEx can benefit from Theano's speed and optimizations, making MCEx's logic very succinct. Theano's graph structure is also very natural for Bayesian computation.