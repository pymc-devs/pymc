|Tests Status| |Coverage| |Gitter|


``aeppl`` provides tools for a[e]PPL written in `Aesara <https://github.com/pymc-devs/aesara>`_.


Features
========
- Convert graphs containing Aesara ``RandomVariable``\s into joint
  log-probability graphs
- Transforms for ``RandomVariable``\s that map constrained support spaces to
  unconstrained spaces (e.g. the extended real numbers), and a rewrite that
  automatically applies these transformations throughout a graph
- Tools for traversing and transforming graphs containing ``RandomVariable``\s
- ``RandomVariable``-aware pretty printing and LaTeX output


Examples
========

Using ``aeppl``, one can create a joint log-probability graph from a graph
containing Aesara ``RandomVariable``\s:

.. code-block:: python

  import aesara
  from aesara import tensor as at

  from aeppl import joint_logprob, pprint


  # A simple scale mixture model
  S_rv = at.random.invgamma(0.5, 0.5)
  Y_rv = at.random.normal(0.0, at.sqrt(S_rv))

  # Compute the joint log-probability
  y = at.scalar("y")
  s = at.scalar("s")
  logprob = joint_logprob({Y_rv: y, S_rv: s})


Log-probability graphs are standard Aesara graphs, so we can compute
values with them:

.. code-block:: python

  logprob_fn = aesara.function([y, s], logprob)

  logprob_fn(-0.5, 1.0)
  # array(-2.46287705)


Graphs can also be pretty printed:

.. code-block:: python

  from aeppl import pprint, latex_pprint


  # Print the original graph
  print(pprint(Y_rv))
  # b ~ invgamma(0.5, 0.5) in R, a ~ N(0.0, sqrt(b)**2) in R
  # a

  print(latex_pprint(Y_rv))
  # \begin{equation}
  #   \begin{gathered}
  #     b \sim \operatorname{invgamma}\left(0.5, 0.5\right)\,  \in \mathbb{R}
  #     \\
  #     a \sim \operatorname{N}\left(0.0, {\sqrt{b}}^{2}\right)\,  \in \mathbb{R}
  #   \end{gathered}
  #   \\
  #   a
  # \end{equation}

  # Simplify the graph so that it's easier to read
  from aesara.graph.rewriting.utils import rewrite_graph
  from aesara.tensor.rewriting.basic import topo_constant_folding


  logprob = rewrite_graph(logprob, custom_rewrite=topo_constant_folding)


  print(pprint(logprob))
  # s in R, y in R
  # (switch(s >= 0.0,
  #         ((-0.9189385175704956 +
  #           switch(s == 0, -inf, (-1.5 * log(s)))) - (0.5 / s)),
  #         -inf) +
  #  ((-0.9189385332046727 + (-0.5 * ((y / sqrt(s)) ** 2))) - log(sqrt(s))))


Joint log-probabilities can be computed for some terms that are *derived* from
``RandomVariable``\s, as well:

.. code-block:: python

  # Create a switching model from a Bernoulli distributed index
  Z_rv = at.random.normal([-100, 100], 1.0, name="Z")
  I_rv = at.random.bernoulli(0.5, name="I")

  M_rv = Z_rv[I_rv]
  M_rv.name = "M"

  z = at.vector("z")
  i = at.lscalar("i")
  m = at.scalar("m")
  # Compute the joint log-probability for the mixture
  logprob = joint_logprob({M_rv: m, Z_rv: z, I_rv: i})


  logprob = rewrite_graph(logprob, custom_rewrite=topo_constant_folding)

  print(pprint(logprob))
  # i in Z, m in R, a in Z
  # (switch((0 <= i and i <= 1), -0.6931472, -inf) +
  #  ((-0.9189385332046727 + (-0.5 * (((m - [-100  100][a]) / [1. 1.][a]) ** 2))) -
  #   log([1. 1.][a])))


Installation
============

The latest release of ``aeppl`` can be installed from PyPI using ``pip``:

::

    pip install aeppl



The current development branch of ``aeppl`` can be installed from GitHub, also using ``pip``:

::

    pip install git+https://github.com/aesara-devs/aeppl



.. |Tests Status| image:: https://github.com/aesara-devs/aeppl/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/aesara-devs/aeppl/actions/workflows/test.yml
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aeppl/branch/main/graph/badge.svg?token=L2i59LsFc0
  :target: https://codecov.io/gh/aesara-devs/aeppl
.. |Gitter| image:: https://badges.gitter.im/aesara-devs/aeppl.svg
   :alt: Join the chat at https://gitter.im/aesara-devs/aeppl
   :target: https://gitter.im/aesara-devs/aeppl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
