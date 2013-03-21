***********************
 PyMC 3 
***********************

PyMC grew out of, MCEx, an experimental package designed to be allow experimentation with MCMC package design. 
It's goal is to be simple to use, understand, extend and improve, while still being fast. 
The hope is that some of the lessons learned in this experimental package lead to improvements
in PyMC. This branch is still experimental so people are encouraged to try out their own designs and improvements 
as well as make criticisms.

Guided Examples: 
 * [Simple model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc/pymc3/examples/tutorial.ipynb)
 * More advanced [Stochastic Volatility model](http://nbviewer.ipython.org/urls/raw.github.com/pymc-devs/pymc/pymc3/examples/stochastic_volatility.ipynb)

Some design decisions

+----------------------------------+---------------------------------------+---------------------------------------------------+
| Design decision                  | Advantages                            | Disadvantages                                     |
+==================================+=======================================+===================================================+
| Computational core outsourced    | - Simple package code (<400 lines)    | - Supporting arbitrary stochastics/deterministics |
| to Theano                        | - Efficient                           |   more difficult in complex cases                 |
|                                  | - Improvements to Theano improve PyMC |                                                   |
|                                  | - GPU enabled                         |                                                   |
|                                  | - Automatic Differentiation           |                                                   |
+----------------------------------+---------------------------------------+---------------------------------------------------+
| Random variables, distributions, | - Easy to understand design           | - More verbose                                    |
| chains, chain history,           | - Reflects the mathematical structure |                                                   |
| and model all distinct           | - Adding new functionality can be     |                                                   |
|                                  |   done independently from the rest of |                                                   |                       
|                                  |   the package                         |                                                   |
+----------------------------------+---------------------------------------+---------------------------------------------------+
| Functional style design          | - Allows easier exchange of           | - Design less similar to Object Oriented design   |      
|                                  |   components                          |                                                   |
+----------------------------------+---------------------------------------+---------------------------------------------------+ 
 
*****
To Do
*****

 * Simplify standard usage. 
 * Build a GPU example 
 * Give `sample` a way of automatically choosing step methods.
 * Do some profiling to see why likelihoods are slower in pymc3 than pymc 
 * Fix step_methods.gibbs.categorical so that it's faster, currently very slow. 
 * Implement a potential object which can take incomplete covariances and exploit the conditional independence of nodes to do the whole calculation 
 * Build examples showcasing different samplers
 * Reconsider nphistory design
 * missing value imputation
 
****************
Possible Changes
****************

 * Make HMC and related automatically choose a variance/covariance
