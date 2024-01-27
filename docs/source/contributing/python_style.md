(python_style)=
# Python style guide

## Pre commit checks

Some code-quality checks are performed during continuous integration. The easiest way to check that they pass locally,
before submitting your pull request, is by using [pre-commit](https://pre-commit.com/).

Steps to get set up are (run these within your virtual environment):

1. install:

    ```bash
    pip install pre-commit
    ```

2. enable:

    ```bash
    pre-commit install
    ```

Now, whenever you stage some file, when you run `git commit -m "<some descriptive message>"`, `pre-commit` will run
the checks defined in `.pre-commit-config.yaml` and will block your commit if any of them fail. If any hook fails, you
should fix it (if necessary), run `git add <files>` again, and then re-run `git commit -m "<some descriptive message>"`.

You can skip `pre-commit` using `--no-verify`, e.g.

```bash
git commit -m "wip lol" --no-verify
```

To skip one particular hook, you can set the `SKIP` environment variable. E.g. (on Linux):

```bash
SKIP=ruff git commit -m "<descriptive message>"
```

You can manually run all `pre-commit` hooks on all files with

```bash
pre-commit run --all-files
```

or, if you just want to manually run them on a subset of files,

```bash
pre-commit run --files <file_1> <file_2> ... <file_n>
```

## Gotchas & Troubleshooting
__Pre-commit runs on staged files__

If you have some `git` changes staged and other unstaged, the `pre-commit` will only run on the staged files.

__Pre-commit repeatedly complains about the same formatting changes__

Check the unstaged changes (see previous point).

__Whitespace changes in the `environment-dev.yml` files__

On Windows, there are some bugs in pre-commit hooks that can lead to changes in some environment YAML files.
Until this is fixed upstream, you should __ignore these changes__.
To actually make the commit, deactivate the automated `pre-commit` with `pre-commit uninstall` and make sure to run it manually with `pre-commit run --all`.

__Failures in the `mypy` step__

We are running static type checks with `mypy` to continuously improve the reliability and type safety of the PyMC codebase.
However, there are many files with unresolved type problems, which is why we are allowing some files to fail the `mypy` check.

If you are seeing the `mypy` step complain, chances are that you are in one of the following two situations:
* 😕 Your changes introduced type problems in a file that was previously free of type problems.
* 🥳 Your changes fixed type problems.

In any case __read the logging output of the `mypy` hook__, because it contains the instructions how to proceed.

You can also run the `mypy` check manually with `python scripts/run_mypy.py [--verbose]`.
