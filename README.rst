|Tests Status| |Coverage|


``aeppl`` provides tools for a[e]PPL written in `Aesara <https://github.com/pymc-devs/aesara>`_.


Features
========
- Convert graphs containing Aesara ``RandomVariable`` into joint log-probability graphs
- Tools for traversing and transforming graphs containing ``RandomVariable``
- ``RandomVariable``-aware pretty printing and LaTeX output


Examples
========

.. code-block:: python

  import aesara
  from aesara import tensor as at

  from aeppl import joint_logprob, pprint


  # A simple scale mixture model
  S_rv = at.random.invgamma(0.5, 0.5)
  Y_rv = at.random.normal(0.0, at.sqrt(S_rv))

  pprint(Y_rv)
  # S ~ invgamma(0.5, 0.5) in R, Y ~ N(0.0, sqrt(S)**2) in R
  # Y


  # Compute the joint log-probability
  y = at.scalar("y")
  s = at.scalar("s")
  logprob = joint_logprob(Y_rv, {Y_rv: y, S_rv: s})


  # Simplify the graph so that it's easier to read
  from aesara.graph.opt_utils import optimize_graph
  from aesara.tensor.basic_opt import topo_constant_folding


  logprob = optimize_graph(logprob, custom_opt=topo_constant_folding)


  print(pprint(logprob))
  # s in R, y in R
  # (switch(s >= 0.0,
  #         ((-0.9189385175704956 +
  #           switch(s == 0, -inf, (-1.5 * log(s)))) - (0.5 / s)),
  #         -inf) +
  #  ((-0.9189385332046727 + (-0.5 * ((y / sqrt(s)) ** 2))) - log(sqrt(s))))


  # Create a finite mixture model with a Bernoulli distributed
  # mixing distribution
  Z_rv = at.random.normal([-100, 100], 1.0, name="Z")
  I_rv = at.random.bernoulli(0.5, name="I")

  M_rv = Z_rv[I_rv]
  M_rv.name = "M"

  z = at.vector("z")
  i = at.lscalar("i")
  m = at.scalar("m")
  # Compute the joint log-probability for the mixture
  logprob = joint_logprob(M_rv, {M_rv: m, Z_rv: z, I_rv: i})


  logprob = optimize_graph(logprob, custom_opt=topo_constant_folding)

  print(pprint(logprob))
  # i in Z, m in R, a in Z
  # (switch((0 <= i and i <= 1), -0.6931472, -inf) +
  #  ((-0.9189385332046727 + (-0.5 * (((m - [-100  100][a]) / [1. 1.][a]) ** 2))) -
  #   log([1. 1.][a])))


.. |Tests Status| image:: https://github.com/aesara-devs/aeppl/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/aesara-devs/aeppl/actions/workflows/test.yml
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aeppl/branch/main/graph/badge.svg?token=L2i59LsFc0
  :target: https://codecov.io/gh/aesara-devs/aeppl
