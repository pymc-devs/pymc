# Pull request step-by-step
The preferred workflow for contributing to PyMC is to fork the [GitHub repository](https://github.com/pymc-devs/pymc/), clone it to your local machine, and develop on a feature branch.

## Steps

1. Fork the [project repository](https://github.com/pymc-devs/pymc/) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the PyMC repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/pymc.git
   $ cd pymc
   $ git remote add upstream git@github.com:pymc-devs/pymc.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```


   :::{attention}
   Always use a ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository.
   :::

4. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``. The easiest (and recommended) way to set up a development environment is via [miniconda](https://docs.conda.io/en/latest/miniconda.html):

   If using Windows:

   ```bash
   conda env create -f .\conda-envs\windows-environment-dev-py38.yml
   conda activate pymc-dev-py38
   pip install -e .
   ```

   For other platforms:

   ```bash
   $ conda env create -f conda-envs/environment-dev-py37.yml  # or py38 or py39
   $ conda activate pymc-dev-py37
   $ pip install -e .
   ```

   _Alternatively_ you may (probably in a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/)) run:

   ```bash
   $ pip install -e .
   $ pip install -r requirements-dev.txt
   ```

   Yet another alternative is to create a docker environment for development. See: [Developing in Docker](#Developing-in-Docker).

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the PyMC repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

   :::{tip}
   Now that your PR is ready, read the {ref}`pr_checklist` to make sure it follows best practices.
   :::
