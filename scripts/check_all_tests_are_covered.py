"""
In .github/workflows/pytest.yml, tests are split between multiple jobs.

Here, we check that the jobs ignored by the first job actually end up getting
run by the other jobs.
This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-no-tests-are-ignored --all`.
"""

from pathlib import Path

import re

if __name__ == "__main__":
    pytest_ci_job = Path(".github") / "workflows/pytest.yml"
    txt = pytest_ci_job.read_text()
    ignored_tests = set(re.findall(r"(?<=--ignore=)(pymc3/tests.*\.py)", txt))
    non_ignored_tests = set(re.findall(r"(?<!--ignore=)(pymc3/tests.*\.py)", txt))
    assert (
        ignored_tests <= non_ignored_tests
    ), f"The following tests are ignored by the first job but not run by the others: {ignored_tests.difference(non_ignored_tests)}"
    assert (
        ignored_tests >= non_ignored_tests
    ), f"The following tests are run by multiple jobs: {non_ignored_tests.difference(ignored_tests)}"
