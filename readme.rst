**********************
Monte-Carlo Experiment
**********************

MCEx is an experimental package designed to be allow experimentation with how an MCMC package should be designed. It's goal is to be simple to use, understand, extend and improve, while still being fast. The hope is that some of the lessons learned in this experimental package lead to improvements in PyMC.

MCEx outsources its computational core to Theano. The advantage of this is that Theano is that MCEx can benefit from Theano's speed and optimizations, making MCEx's logic very succinct. Theano's graph structure is also very natural for Bayesian computation.

 * Free variables separate from prior distributions. This means all variables are like Potentials. Prior distributions must be added explicitly. Advantages:
   * it is simple and obvious (operationally and conceptually) how to 
   * computational framework represents computational graph quite closely
  Disadvatages: 
   * more verbose
 * free variables, chains, chain history, and the model are all very distinct objects
 * objects are constructed explicitly instead of implicitly. Advantages: structure of the package is more obvious. Disadvantages: package is more verbose
 * 