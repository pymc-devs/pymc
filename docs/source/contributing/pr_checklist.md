(pr_checklist)=
# Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, use the pull request title to describe the issue and mention the issue number in the pull request _description_.
   This will make sure a link back to the original issue is created.

   :::{caution}
   Adding the related issue in the PR title generates no link and is therefore
   not useful as nobody knows issue numbers. Please mention all related
   issues in the PR but do so only in the PR description.
   :::

*  All public methods must have informative docstrings with sample usage when appropriate.
   Docstrings should follow the [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html)

*  Please select "Create draft pull request" in the dropdown menu when opening your pull request to indicate a work in progress. This is to avoid duplicated work, to get early input on implementation details or API/functionality, or to seek collaborators.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.
*  When adding additional functionality, consider adding also one example notebook at [pymc-examples](https://github.com/pymc-devs/pymc-examples).
   Open a [proposal issue](https://github.com/pymc-devs/pymc-examples/issues/new/choose) in the example repo to discuss the specific scope of the notebook.

* Run any of the pre-existing examples in [pymc-examples](https://github.com/pymc-devs/pymc-examples) that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit test, but also to show how your contribution improves PyMC for end users.

* **No `pre-commit` errors:** see the {ref}`python_style` and {ref}`jupyter_style` page on how to install and run it.

*  All other tests pass when everything is rebuilt from scratch. See {ref}`running_the_test_suite`
