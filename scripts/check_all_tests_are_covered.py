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
