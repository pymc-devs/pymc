(installation)=
# Installation

To install PyMC, select the operating system where you want to perform the installation.

## Linux and Mac OS
We recommend using [Anaconda](https://www.anaconda.com/) (or [Miniforge](https://github.com/conda-forge/miniforge)) to install Python on MacOS, which allows for packages to be installed using its `conda` utility. We recommend installing PyMC into a clean `conda` environment, which here is called `pymc_env` but can be changed to anything else:

```console
conda create -n pymc_env pymc python pip
```

YOu can also use pip. Activate the environment that you wish to install into, and install as follows:

```console
conda activate pymc_env
pip install pymc --pre
```

### JAX sampling

If you wish to enable sampling using the JAX backend via NumPyro (experimental), the following should also be installed:

```console
pip install jax jaxlib numpyro
```

Similarly, to use BlackJAX for sampling it should be installed via `pip`:

```console
pip install blackjax
```

### MKL (Math Kernel Library)
If you have an Intel chip, you are strongly suggested to install MKL

```console
conda create -c conda-forge mkl mkl-service
```


## Windows

The following instructions rely on having [Anaconda](https://www.anaconda.com/products/individual), [Mamba](https://github.com/mamba-org/mamba) or [Miniforge](https://github.com/conda-forge/miniforge) installed, which provide Python environments from which you can install and run PyMC in a controlled way, using the `conda` utility.

### Simple Install

The Simple Install process of PyMC under Windows is recommended for most users. It is a two-step process, with `conda` being used to set up an environment that contains required dependencies, including the GCC compiler toolchain, and then `pip` to install PyMC itself. Using `conda` allows Aesara and PyMC to easily access MKL and also confines the install of the GCC compiler toolchain into the `conda` environment rather than placing it onto the global Windows `PATH`.

It is usually a good idea to install into a fresh `conda` environment, which we will call `pymc_env`:

```console
conda create -n pymc_env -c conda-forge python libpython numba m2w64-toolchain
```

Once the `conda` environment has been created, there are two options for getting a version of PyMC installed depending on your needs. You probably want Option 1, which installs the latest version of PyMC. But Option 2 is available if you know you need the older PyMC3.

Next, PyMC can be installed into the environment we have created:

```console
conda activate pymc_env
pip install pymc --pre
```

Until v4 is officially released, you will need to add the `--pre` flag as shown to get the correct version.

You might experience this warning message when running PyMC:

    WARNING (aesara.tensor.blas): Using NumPy C-API based implementation for BLAS functions.

If so, create a file called `.aesararc`, and put it in your home directory (usually `C:\Users\<username>`). Add these lines to `.aesararc`:

 ```
[blas]
ldflags = -lblas
```

That will signal Aesara to link against BLAS, and that should eliminate the warning message.


### Fancy Install

The Fancy Install of PyMC is for those who would prefer to manage the GCC compiler installation on their own rather than rely on `conda` to do it. You can install an up-to-date copy of GCC yourself and make sure it is available on the global Windows `PATH`. An easy way to do this (though not the only way) is via the [Chocolatey](https://chocolatey.org/install) package manager:

```console
choco install mingw
```

Once GCC installation has completed, you can then pickup the creation of a `conda` environment and the PyMC or PyMC3 install options as described above, replacing the `conda` environment creation command with this one, which omits the `m2w64-toolchain`:

```console
conda create -n pymc_env -c conda-forge python libpython numba
```

Then follow conda instruction above.

### MKL (Math Kernel Library)
If you have an Intel chip, you are strongly suggested to install MKL

```console
conda create -c conda-forge mkl mkl-service
```

### JAX sampling

JAX is not directly supported on Windows systems at the moment.
