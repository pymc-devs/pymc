# PyMC 3 Changelog

This is a list of important changes to PyMC 3 relative to the 2.x releases.

* Added gradient-based MCMC samplers: Hamiltonian MC (`HMC`) and No-U-Turn Sampler (`NUTS`)
* Added [slice sampler](http://projecteuclid.org/DPubS?verb=Display&version=1.0&service=UI&handle=euclid.aos/1056562461&page=record)
* Automatic gradient calculations using [Theano](https://github.com/Theano/Theano)
* Convenient generalized linear model specification using [Patsy](http://patsy.readthedocs.org/en/latest/) formulae
* Parallel sampling via `multiprocessing` (IPython parallel support planned)
* New model specification using context managers