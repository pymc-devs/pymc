***********************
 Monte-Carlo Experiment
***********************

MCEx is an experimental package designed to be allow experimentation with MCMC package design. 
It's goal is to be simple to use, understand, extend and improve, while still being fast. 
The hope is that some of the lessons learned in this experimental package lead to improvements
in PyMC. People are encouraged to fork MCEx to try out their own designs and improvements 
as well as make criticisms.

For a tutorial on basic inference, see tutorial.py in the examples folder.

Some design decisions

+----------------------------------+---------------------------------------+---------------------------------------------------+
| Design decision                  | Advantages                            | Disadvantages                                     |
+==================================+=======================================+===================================================+
| Computational core outsourced    | - Simple package code (<400 lines)    | - Supporting arbitrary stochastics/deterministics |
| to Theano                        | - Efficient                           |   more difficult in complex cases                 |
|                                  | - Improvements to Theano improve MCEx |                                                   |
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
 
****
TODO
****

 * Build a GPU example 
 * Make nphistory a default for sample() so that you don't have to construct it every time
 * Do some profiling to see why likelihoods are slower in mcex than pymc 
 * Fix step_methods.gibbs.categorical so that it's faster, currently very slow. 
 * Implement a potential object which can take incomplete covariances and exploit the conditional independence of nodes to do the whole calculation 
 * Build examples showcasing different samplers
 * Reconsider nphistory design