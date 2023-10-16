(running_the_test_suite)=
# Running the test suite
The first step to run tests is the installation of additional dependencies that are needed for testing:

```bash
pip install -r requirements-dev.txt
```

The PyMC test suite uses `pytest` as the testing framework.
If you are unfamiliar with `pytest`, check out [this short video series](https://calmcode.io/pytest/introduction.html).

With the optional dependencies installed, you can start running tests.
Below are some example of how you might want to run certain parts of the test suite.

```{attention}
Running the entire test suite will take hours.
Therefore, we recommend to run just specific tests that target the parts of the codebase you're working on.
```

To run all tests from a single file:
```bash
pytest -v tests/model/test_core.py
```

```{tip}
The `-v` flag is short-hand for `--verbose` and prints the names of the test cases that are currently running.
```

Often, you'll want to focus on just a few test cases first.
By using the `-k` flag, you can filter for test cases that match a certain pattern.
For example, the following command runs all test cases from `test_core.py` that have "coord" in their name:

```bash
pytest -v tests/model/test_core.py -k coord
```


To get a coverage report, you can pass `--cov=pymc`, optionally with `--cov-report term-missing` to get a printout of the line numbers that were visited by the invoked tests.
Note that because you are not running the entire test suite, the coverage will be terrible.
But you can still watch for specific line numbers of the code that you're working on.

```bash
pytest -v --cov=pymc --cov-report term-missing tests/<name of test>.py
```

When you are reasonably confident about the changes you made, you can push the changes and open a pull request.
Our GitHub Actions pipeline will run the entire test suite and if there are failures you can go back and run these tests on your local machine.
