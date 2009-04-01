**********
Conclusion
**********

MCMC is a surprisingly difficult and bug-prone algorithm to implement by hand.
We find PyMC makes it much easier and less stressful. PyMC also makes our work
more dynamic; getting hand-coded MCMC's working used to be so much work that we
were reluctant to change anything, but with PyMC changing models is a breeze. We
hope it does the same for you!



What's next?
============

A partial list of the features we would like to include in future releases
follows. Three stars means that only debugging and checking is needed, so the
feature is likely to be available in release 2.1; two stars means that there are
no conceptual hurdles to be overcome but there's a lot of work left to do; and
one star means only experimental development has been done.

* (\*\*\*) Gibbs step methods to handle conjugate and nearly-conjugate
  submodels,

* (\*\*\*) Handling all-normal submodels with sparse linear algebra
  [Wilkinson:2004]_. This will help PyMC handle Markov random fields and time-
  series models based on the 'dynamic linear model' [West:1997]_ (among others)
  more efficiently, in some cases fitting them in closed form,

* (\*\*) Generic Monte Carlo EM and SEM algorithms,

* (\*\*) Parallelizing single chains with a thread pool,

* (\*\*) Terse syntax inspired by [Kerman:2004]_ for creating variables: ``C=A*B``
  should return a deterministic object if :math:`A` and/or :math:`B` is a PyMC
  variable,

* (\*\*) Distributing multiple chains across multiple processes,

* (\*) Parsers for model-definition syntax from R and WinBugs,

* (\*) Dirichlet processes and other stick-breaking processes [Ishwaran:2001]_.

These features will make their way into future releases as (and if) we are able
to finish them and make them reliable.


How to get involved
===================

We welcome new contributors at all levels. If you would like to contribute to
any of the features above, or to improve PyMC itself in some other way, please
introduce yourself on our `mailing list`_. If you
would like to share code written in PyMC, for example a tutorial or a
specialized step method, please feel free to edit our `wiki page`_.

.. _`mailing list`: pymc@googlegroups.com

.. _`wiki page`: http://code.google.com/p/pymc/w/list

