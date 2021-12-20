(pr_checklist)=
# Pull request checklist

We strongly recommended that all contribution comply with the following guidelines before being merged:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

   :::{caution}
   Adding the related issue in the PR title generates no link and is therefore
   not useful as nobody knows issue numbers. Please mention all related
   issues in the PR but do so only in the PR description.
   :::

*  All public methods must have informative docstrings with sample usage when appropriate.

*  Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request broad review of functionality or API, or (3) seek collaborators.

*  All other tests pass when everything is rebuilt from scratch. See {ref}`running_the_test_suite`

*  When adding additional functionality, consider adding also one example notebook at [pymc-examples](https://github.com/pymc-devs/pymc-examples). Open a [proposal issue](https://github.com/pymc-devs/pymc-examples/issues/new/choose) in the example repo to discuss the specific scope of the notebook.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Run any of the pre-existing examples in [pymc-examples](https://github.com/pymc-devs/pymc-examples) that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves PyMC for end users.

You can also check for common programming errors with the following
tools:

* Code with good test **coverage** (at least 80%), check with:

  ```bash
  $ pip install pytest pytest-cov coverage
  $ pytest --cov=pymc pymc/tests/<name of test>.py
  ```

* No `pre-commit` errors: see the {ref}`python_style` and {ref}`jupyter_style` page on how to install and run it.
