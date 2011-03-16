**********************
Monte-Carlo Experiment
**********************

MCEx is an experimental package designed to be allow experimentation with how an MCMC package should be designed. It's goal is to be simple to use, understand, extend and improve, while still being fast. The hope is that some of the lessons learned in this experimental package lead to improvements in PyMC.

 - Some design decision: computational core is outsourced to Theano.
 - Free variables separate from prior distributions. This means all variables are like Potentials. Prior distributions must be added explicitly. Say you want to fit a model like f(x) ~ Norm(0,1) where x is your data and f is a transformation that depends on parameters. It is not straightforward to fit this in PyMC, but it is in MCEx.
 - Free variables, chains, chain history, and the model are all very distinct objects
 - Objects are constructed more explicitly by the user 

 +----------------------------------+---------------------------------------+---------------------------------------------------+
 | Design feature                   | Advantages                            | Disadvantages                                     |
 +==================================+=======================================+===================================================+
 | Computational core out-sourced   | - simple package code                 | - supporting arbitrary stochastics/deterministics |
 | to Theano                        | - efficient                           |    more difficult in complex cases                |
 |                                  | - improvements to Theano improve MCEx |                                                   |
 +----------------------------------+---------------------------------------+---------------------------------------------------+
 | Free variables separate from     | - obvious how to apply unknown        | - more verbose                                    |
 | prior distributions              |     transformations to the data       |                                                   |
 |                                  | - represents computational            |                                                   |
 |                                  |    graph closely                      |                                                   |
 +----------------------------------+---------------------------------------+---------------------------------------------------+
 | free variables, chains,          | - easy to understand design           |                                                   |
 | chain history, and model all     | - hopefully robust                    |                                                   |
 | distinct                         |                                       |                                                   |
 +----------------------------------+---------------------------------------+---------------------------------------------------+
 | objects constructed              | - more obvious to user                | - more verbose                                    |
 | explicitly by user               |   how package works                   |                                                   |
 +----------------------------------+---------------------------------------+---------------------------------------------------+
