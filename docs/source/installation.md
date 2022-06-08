(installation)=
# Installation

To install PyMC, select the operating system where you want to perform the installation.

## Linux

We recommend using [Anaconda](https://www.anaconda.com/) (or [Miniforge](https://github.com/conda-forge/miniforge)) to install Python on MacOS, which allows for packages to be installed using its `conda` utility.

```console
conda create -c conda-forge -n pymc_env pymc
conda activate pymc_env
```
If you like, replace the name `pymc_env` with whatever environment name you prefer.

### JAX sampling

If you wish to enable sampling using the JAX backend via NumPyro (experimental), the following should also be installed:

```console
pip install jax jaxlib numpyro
```

Similarly, to use BlackJAX for sampling it should be installed via `pip`:

```console
pip install blackjax
```

### PyMC3 Installation

If you are looking for PyMC3, it can be installed from Conda Forge (conda-forge):

```console
conda create -c conda-forge -n pymc3_env pymc3 theano-pymc mkl mkl-service
conda activate pymc3_env
```

Note that you **must** specifically request `theano-pymc` or you will get an obsolete version of PyMC3 that works with the now-abandoned `theano` library.  We encourage you to test this with the `--dry-run` flag to ensure you get up-to-date versions. 

While discouraged due to reports of installation problems you could try to install PyMC3 and its dependencies via PyPI using `pip`:

```console
pip install pymc3
```

## MacOS

We recommend using [Anaconda](https://www.anaconda.com/) (or [Miniforge](https://github.com/conda-forge/miniforge)) to install Python on MacOS, which allows for packages to be installed using its `conda` utility.

```bash
conda create -c conda-forge -n pymc_env pymc
conda activate pymc_env
```
If you like, replace the name `pymc_env` with whatever environment name you prefer.

### JAX sampling

If you wish to enable sampling using the JAX backend via NumPyro (experimental), the following should also be installed:

```console
pip install jax jaxlib numpyro
```

Similarly, to use BlackJAX for sampling it should be installed via `pip`:

```console
pip install blackjax
```


### PyMC3 installation

If you are looking for PyMC3, then replace the above with:

```console
conda create -c conda-forge -n pymc3_env pymc3
conda activate pymc3_env
```
For older (Intel) Macs, you can should also install the Intel Math Kernel Library (MKL) for improved speed:

```console
conda create -c conda-forge -n pymc3_env python pymc3 theano-pymc mkl mkl-service
conda activate pymc3_env
```

Note that you **must** specifically request `theano-pymc` or you will get an obsolete version of PyMC3 that works with the now-abandoned `theano` library.  We encourage you to test this with the `--dry-run` flag to ensure you get up-to-date versions. 

## Windows

The following instructions rely on having [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86_64.exe), [Mamba](https://github.com/mamba-org/mamba) or [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe) installed, which provide Python environments from which you can install and run PyMC in a controlled way, using the `conda` utility.

### Simple Install

The Simple Install process for PyMC under Windows is recommended for most users, with `conda` being used to set up an environment that contains required dependencies, including the GCC compiler toolchain. Using `conda` allows Aesara and PyMC to easily access MKL and also confines the install of the GCC compiler toolchain into the `conda` environment rather than placing it onto the global Windows `PATH`.

It is usually a good idea to install into a fresh `conda` environment, which we will call `pymc_env`:

```console
conda create -n pymc_env -c conda-forge pymc
```

Next, you can activate the environment in which you just installed PyMC.

```console
conda activate pymc_env
```

### Fancy Install

The Fancy Install of PyMC is for those who would prefer to manage the GCC compiler installation on their own rather than rely on `conda` to do it. You can install an up-to-date copy of GCC yourself and make sure it is available on the global Windows `PATH`. An easy way to do this (though not the only way) is via the [Chocolatey](https://chocolatey.org/install) package manager:

```console
choco install mingw
```
Once GCC installation has completed, you can then pickup the creation of a `conda` environment as described above, replacing the `conda` environment creation command with this one, which omits the `m2w64-toolchain`:

```console
conda create -n pymc_env -c conda-forge pymc-base libpython mkl-service numba
```

### JAX sampling

JAX is not directly supported on Windows systems at the moment.

### PyMC3 installation

If you are looking for PyMC3, then replace the above with:

```console
conda create -c conda-forge -n pymc3_env pymc3
conda activate pymc3_env
```
