(pr_tutorial)=
# Pull request step-by-step
The preferred workflow for contributing to PyMC is to fork the [GitHub repository](https://github.com/pymc-devs/pymc/), clone it to your local machine, and develop on a feature branch.

## Steps

1. Read the {ref}`pr_etiquette`.

1. Fork the [project repository](https://github.com/pymc-devs/pymc/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

1. Clone your fork of the PyMC repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your GitHub handle>/pymc.git
   cd pymc
   git remote add upstream git@github.com:pymc-devs/pymc.git
   ```

1. Create a ``feature`` branch to hold your development changes:

   ```bash
   git checkout -b my-feature
   ```


   :::{attention}
   Always use a ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository.
   :::

1. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``.
   The easiest (and recommended) way to set up a development environment is via [miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ::::{tab-set}
   :::{tab-item} Linux/MacOS

   ```bash
   conda env create -f conda-envs/environment-dev.yml
   ```
   :::
   :::{tab-item} Windows
   ```bash
   conda env create -f .\conda-envs\windows-environment-dev.yml
   ```
   :::
   ::::

   ```bash
   conda activate pymc-dev
   pip install -e .
   ```

   _Alternatively_ you may (probably in a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/)) run:

   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

  <!-- Commented out because our Docker image is outdated/broken.
    Yet another alternative is to create a docker environment for development. See: [Developing in Docker](#Developing-in-Docker).
  -->

1. Develop the feature on your feature branch.

   ```bash
   git checkout my-feature   # no -b flag because the branch is already created
   ```

1. Before committing, run `pre-commit` checks.

   ```bash
   pip install pre-commit
   pre-commit run --all      # 👈 to run it manually
   pre-commit install        # 👈 to run it automatically before each commit
   ```

1. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.

1. After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Then push the changes to the fork in your GitHub account with:

   ```bash
   git push -u origin my-feature
   ```

   :::{sidebar}
   If this is your first contribution, the start of some CI jobs will have to be approved by a maintainer.
   :::

1. Go to the GitHub web page of your fork of the PyMC repo.
   Click the 'Pull request' button to send your changes to the project's maintainers for review.
   This will send a notification to the committers.

   :::{tip}
   Now that your PR is ready, read the {ref}`pr_checklist` to make sure it follows best practices.
   :::
