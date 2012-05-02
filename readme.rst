***********************
 Monte-Carlo Experiment
***********************

MCEx is an experimental package designed to be allow experimentation with MCMC package design. 
It's goal is to be simple to use, understand, extend and improve, while still being fast. 
The hope is that some of the lessons learned in this experimental package lead to improvements
in PyMC. People are encouraged to fork MCEx to try out their own designs and improvements 
as well as make criticisms.

Some design decisions
+----------------------------------+---------------------------------------+---------------------------------------------------+
| Design decision                  | Advantages                            | Disadvantages                                     |
+==================================+=======================================+===================================================+
| Computational core outsourced    | - Simple package code (~400 lines)    | - supporting arbitrary stochastics/deterministics |
| to Theano                        | - Efficient                           |   more difficult in complex cases                 |
|                                  | - Improvements to Theano improve MCEx |                                                   |
|                                  | - GPU enabled                         |                                                   |
|                                  | - Automatic Differentiation           |                                                   |
+----------------------------------+---------------------------------------+---------------------------------------------------+
| Random variables, chains,        | - Easy to understand design           |  - more verbose                                   |
| chain history, and model all     | - Reflects the mathematical structure |                                                   |
| distinct                         | - Adding new functionality can be done|                                                   |
|                                  |   independently from the rest of the  |                                                   |                       
|                                  |   package                             |                                                   |
+----------------------------------+---------------------------------------+---------------------------------------------------+
 
 