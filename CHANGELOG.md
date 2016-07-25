# PyMC 3 Changelog

This is a list of important changes to PyMC 3 relative to the 2.x releases.

* Added gradient-based MCMC samplers: Hamiltonian MC (`HMC`) and No-U-Turn Sampler (`NUTS`)
* Added [slice sampler](http://projecteuclid.org/DPubS?verb=Display&version=1.0&service=UI&handle=euclid.aos/1056562461&page=record)
* Automatic gradient calculations using [Theano](https://github.com/Theano/Theano)
* Convenient generalized linear model specification using [Patsy](http://patsy.readthedocs.org/en/latest/) formulae
* Parallel sampling via `multiprocessing` (IPython parallel support planned)
* New model specification using context managers
* New Automatic Differentiation Variational Inference[AVDI] (http://arxiv.org/abs/1506.03431) (`ADVI`) allowing faster sampling than `HMC` for some problems.
* Model evaluation like Deviance Information Critertion `DIC` and `WAIC`
* Numerous docs, examples and API documentation highlighting the power of this model specification API. 