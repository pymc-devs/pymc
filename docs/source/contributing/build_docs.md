# Build documentation locally

:::{warning}
Docs build is not supported on Windows.
To build docs on Windows we recommend running inside a Docker container.
:::

To build the docs, run these commands at PyMC repository root:

## Installing dependencies

```shell
conda install -f conda-envs/environment-docs.yml  # or make sure all dependencies listed here are installed
pip install -e .  # Install local pymc version as installable package
```

## Building the documentation
There is a `Makefile` in the pymc repo to help with the doc building process.

```shell
make clean
make html
```

`make html` is the command that builds the documentation with `sphinx-build`.
`make clean` deletes caches and intermediate files.

The `make clean` step is not always necessary, if you are working on a specific page
for example, you can rebuild the docs without the clean step and everything should
work fine. If you are restructuring the content or editing toctrees, then you'll need
to execute `make clean`.

A good approach is to generally skip the `make clean`, which makes
the `make html` faster and see how everything looks.
If something looks strange, run `make clean` and `make html` one after the other
to see if it fixes the issue before checking anything else.

### Emulate building on readthedocs
The target `rtd` is also available to chain `make clean` with `sphinx-build`
setting also some extra options and environment variables to indicate
sphinx to simulate as much as possible a readthedocs build.

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
