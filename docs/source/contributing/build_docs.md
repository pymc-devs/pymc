# Build documentation locally

:::{warning}
Docs build is not supported on Windows.
To build docs on Windows we recommend running inside a Docker container.
:::

To build the docs, first install dependencies by running these commands at the PyMC repo root:

```shell
# create the pymc-docs conda env, or equivalently make
# sure all dependencies listed in this file are installed
conda env create -f conda-envs/environment-docs.yml

# Install local pymc version in editable mode
pip install -e .
```

## Jupyter kernel for notebook builds

The documentation configuration uses the `python3` Jupyter kernel for notebooks.
Make sure this kernel points to the same environment where you installed PyMC in
editable mode. To inspect the registered kernels and their locations, run:

```shell
jupyter kernelspec list
```

If the `python3` kernel is missing or points to a different environment, activate
the documentation environment and register its Python executable as that kernel:

```shell
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

This replaces an existing `python3` kernelspec with one for the active environment.

## Building the documentation
There is a `Makefile` in the pymc repo to help with the doc building process.

```shell
make clean
make html
```

`make html` is the command that builds the documentation with `sphinx-build`.
`make clean` deletes caches and intermediate files.

The `make clean` step is not always necessary. If you are working on a specific page,
for example, then you can rebuild the docs without the `clean` step, and everything should
work fine. If you are restructuring the content or editing toctrees, then you'll need
to execute `make clean`.

A good approach is generally to skip `make clean`, which makes
the `make html` faster, and see how everything looks. If something
looks strange, run `make clean` and `make html` one after the other
to see if it fixes the issue before checking anything else.

### Emulate building on readthedocs
The target `rtd` is also available to chain `make clean` with `sphinx-build`
setting also some extra options and environment variables to instruct
sphinx to simulate a readthedocs build as much as possible.

```shell
make rtd
```

:::{important}
This won't reinstall or update any dependencies, unlike on readthedocs where
all dependencies are installed in a clean env before each build.

But it will execute all notebooks inside the `core_notebooks` folder,
which by default are not executed. Executing the notebooks will add several minutes
to the doc build, as there are 6 notebooks which take between 20s to 5 minutes
to run.
:::

## View the generated docs

```shell
make view
```

This will use Python's `webbrowser` module to open the generated website on your browser.
The generated website is static, so there is no need to set a server to preview it.
