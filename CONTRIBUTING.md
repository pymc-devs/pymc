# Guidelines for Contributing

As a scientific community-driven software project, PyMC welcomes contributions from interested individuals or groups. These guidelines are provided to give potential contributors information to make their contribution compliant with the conventions of the PyMC project, and maximize the probability of such contributions to be merged as quickly and efficiently as possible.

There are 4 main ways of contributing to the PyMC project (in descending order of difficulty or scope):

* Adding new or improved functionality to the existing codebase
* Fixing outstanding issues (bugs) with the existing codebase. They range from low-level software bugs to higher-level design problems.
* Contributing or improving the documentation (`docs`) or examples (`pymc/examples`)
* Submitting issues related to bugs or desired enhancements

## Opening issues

We appreciate being notified of problems with the existing PyMC code. We prefer that issues be filed the on [Github Issue Tracker](https://github.com/pymc-devs/pymc/issues), rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for key words in the project issue tracker.

Filter on the ["beginner friendly"](https://github.com/pymc-devs/pymc/issues?q=is%3Aopen+is%3Aissue+label%3A%22beginner+friendly%22) label for issues which are good for new contributors.

## Etiquette for code contributions
* When you start working working on an issue, open a `Draft` pull request as soon as you make your first commit (see steps below).
* Before opening a PR with a new feature, please make a proposal by opening an [issue](https://github.com/pymc-devs/pymc/issues) or [Discussion](https://github.com/pymc-devs/pymc/discussions) with the maintainers. Depending on the proposal we might direct you to other places such as [`pymc-experimental`](https://github.com/pymc-devs/pymc-experimental) or [`pymc-examples`](https://github.com/pymc-devs/pymc-examples).
* Any issue without an open pull request is available for work.
   * If a pull request has no recent activity it may be closed, or taken over by someone else.
   * The specific timeframe for "recent" is hard to define as it depends on the contributor the specific code change, and other contextual factors. As a rule of thumb in a normal pull request with no other blockers there is typically activity every couple of days.
   * The core devs will make their best judgement when opting to close PRs or reassign them to others.
* If unsure if an issue ticket is available feel free to ask in the issue ticket. Note however, that per the previous point an open pull request is way to claim an issue ticket. Please do not make unrealistic pledges in the issue tickets.
* It's okay if you are delayed or need to take a break, but please leave a comment in the pull request if you cannot get it to a state where it can be merged. Depending on the change (urgent bugfix vs. new feature) the core devs can determine if the PR needs to be reassigned to get the work done.

## Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are inclined to do so to submit patches for new or existing issues via pull requests. This is particularly the case for simple fixes, such as typos or tweaks to documentation, which do not require a heavy investment of time and attention.

Contributors are also encouraged to contribute new code to enhance PyMC's functionality, also via pull requests. Please consult the [PyMC documentation](https://pymc-devs.github.io/pymc/) to ensure that any new contribution does not strongly overlap with existing functionality.

The preferred workflow for contributing to PyMC is to fork the [GitHub repository](https://github.com/pymc-devs/pymc/), clone it to your local machine, and develop on a feature branch.

### Steps

1. Read the [Etiquette for code contributions](#etiquette-for-code-contributions).

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

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository.

1. Project requirements are in ``requirements.txt``, and libraries used for development are in ``requirements-dev.txt``. The easiest (and recommended) way to set up a development environment is via [miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   conda env create -f conda-envs/environment-dev-py39.yml  # or py38 or py37
   conda activate pymc-dev-py39
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
   git checkout -b my-cool-bugfix
   ```

1. Before committing, please run `pre-commit` checks.
   ```bash
   pip install pre-commit
   pre-commit run --all      # 👈 to run it manually
   pre-commit install        # 👈 to run it automatically before each commit
   ```

1. Add changed files using ``git add`` and then ``git commit`` files:
   ```bash
   git checkout -b my-cool-bugfix
   git add modified_files
   git commit
   ```

   to record your changes locally.

1. After committing, sync with the base repository in case there have been any changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Then push the changes to the fork in your GitHub account with:

   ```bash
   git push -u origin my-cool-bugfix
   ```

1. Go to the GitHub web page of your fork of the PyMC repo. Click the 'Pull request' button to open a pull request to the main project. Our CI pipeline will start running tests* and project maintainers can start reviewing your changes.
   <sup>*If this is your first contribution, the start of some CI jobs will have to be approved by a maintainer.</sup>

### Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  All public methods must have informative docstrings with sample usage when appropriate.

*  Please select "Create draft pull request" in the dropdown menu when opening your pull request to indicate a work in progress. This is to avoid duplicated work, to get early input on implementation details or API/functionality, or to seek collaborators.

<!-- Commented out because our Docker image is outdated/broken.
See [Developing in Docker](#Developing-in-Docker) for information on running the test suite locally.
-->

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

*  Depending on the functionality, please consider contributing an example Jupyter Notebook to [`pymc-examples`](https://github.com/pymc-devs/pymc-examples). Examples should demonstrate why the new functionality is useful in practice and, if possible, compare it to other methods available in PyMC.

* __No `pre-commit` errors:__ see the [Python code style](https://github.com/pymc-devs/pymc/wiki/Python-Code-Style) and [Jupyter Notebook style](https://github.com/pymc-devs/pymc/wiki/PyMC-Jupyter-Notebook-Style-Guide) page from our Wiki on how to install and run it.

In addition to running `pre-commit`, please also run tests:

```bash
pip install pytest pytest-cov coverage

# To run a subset of tests
pytest --verbose pymc/tests/<name of test>.py

# To get a coverage report
pytest --verbose --cov=pymc --cov-report term-missing pymc/tests/<name of test>.py
```

<!-- Commented out because our Docker image is outdated/broken.
## Developing in Docker

We have provided a Dockerfile which helps for isolating build problems, and local development.
Install [Docker](https://www.docker.com/) for your operating system, clone this repo, then
run `./scripts/start_container.sh`. This should start a local docker container called `pymc`,
as well as a [`jupyter`](http://jupyter.org/) notebook server running on port 8888. The
notebook should be opened in your browser automatically (you can disable this by passing
`--no-browser`). The repo will be running the code from your local copy of `pymc`,
so it is good for development.

You may also use it to run the test suite, with

```bash
$  docker exec -it pymc  bash # logon to the container
$  cd ~/pymc/tests
$  . ./../../scripts/test.sh # takes a while!
```

This should be quite close to how the tests run on TravisCI.

If the container was started without opening the browser, you
need the notebook instances token to work with the notebook. This token can be
accessed with

```
docker exec -it pymc jupyter notebook list
```
-->

## Style guide

We have configured a pre-commit hook that checks for `black`-compliant code style.
We encourage you to configure the pre-commit hook as described in the [PyMC Python Code Style Wiki Page](https://docs.pymc.io/en/latest/contributing/python_style.html), because it will automatically enforce the code style on your commits.

Similarly, consult the [PyMC's Jupyter Notebook Style](https://docs.pymc.io/en/latest/contributing/jupyter_style.html) guides for notebooks.

For documentation strings, we use [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) to comply with the style that predominates in our upstream dependencies.

__This guide was derived from the [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md)__
