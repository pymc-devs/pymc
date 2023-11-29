(installation)=
# Installation

We recommend using [Anaconda](https://www.anaconda.com/) (or [Miniforge](https://github.com/conda-forge/miniforge)) to install Python on your local machine, which allows for packages to be installed using its `conda` utility.

Once you have installed one of the above, PyMC can be installed into a new conda environment as follows:

```console
conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
```
If you like, replace the name `pymc_env` with whatever environment name you prefer.

:::{seealso}
The [conda-forge tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#using-multiple-channels) page to avoid installation
issues when using multiple conda channels (e.g. defaults and conda-forge).
:::

## JAX sampling

If you wish to enable sampling using the JAX backend via NumPyro,
you need to install it manually as it is an optional dependency:

```console
conda install numpyro
```

Similarly, to use BlackJAX sampler instead:

```console
conda install blackjax
```

## Nutpie sampling

You can also enable sampling with [nutpie](https://github.com/pymc-devs/nutpie).
Nutpie uses numba as the compiler and a sampler written in Rust for faster performance.

```console
conda install -c conda-forge nutpie
```
