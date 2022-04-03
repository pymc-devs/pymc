(running_the_test_suite)=
# Running the test suite

TODO: explain steps to run test suite. This is a how to guide, so assume readers know how to install
things and so on, at most mention what dependencies are needed.

```bash
pip install pytest pytest-cov coverage

# To run a subset of tests
pytest --verbose pymc/tests/<name of test>.py

# To get a coverage report
pytest --verbose --cov=pymc --cov-report term-missing pymc/tests/<name of test>.py
```
