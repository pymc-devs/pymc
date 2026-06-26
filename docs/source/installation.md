(installation)=
# Installation

PyMC can be installed with `pip`.

We also recommend installing [nutpie](https://github.com/pymc-devs/nutpie), a
fast NUTS sampler written in Rust. When nutpie is installed, PyMC selects it as
the default NUTS sampler automatically.

```console
pip install "pymc[nutpie]"
```

To install PyMC on its own:

```console
pip install pymc
```

## Installing with conda

As an alternative, PyMC can be installed with `conda` using
[Anaconda](https://www.anaconda.com/) or
[Miniforge](https://github.com/conda-forge/miniforge). This is recommended if you
want to use the legacy C backend or need control over the BLAS library.

We recommend a fresh
conda environment:

```console
conda create -c conda-forge -n pymc_env "pymc>=6"
conda activate pymc_env
```

If you like, replace the name `pymc_env` with whatever environment name you prefer.

To enable nutpie when using conda:

```console
conda install -c conda-forge nutpie
```

:::{seealso}
The [conda-forge tips & tricks](https://conda-forge.org/docs/user/tipsandtricks.html#using-multiple-channels) page to avoid installation
issues when using multiple conda channels (e.g. defaults and conda-forge).
:::

## JAX sampling

If you wish to enable sampling using the JAX backend via NumPyro,
you need to install it manually as it is an optional dependency:

```console
pip install numpyro
```

Similarly, to use the BlackJAX sampler instead:

```console
pip install blackjax
```
