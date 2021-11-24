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
SKIP=pyupgrade git commit -m "<descriptive message>"
```

You can manually run all `pre-commit` hooks on all files with

```bash
pre-commit run --all-files
```

or, if you just want to manually run them on a subset of files,

```bash
pre-commit run --files <file_1> <file_2> ... <file_n>
```
