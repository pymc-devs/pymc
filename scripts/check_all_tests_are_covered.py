"""
In .github/workflows/pytest.yml, tests are split between multiple jobs.

Here, we check that the jobs ignored by the first job actually end up getting
run by the other jobs.
This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-no-tests-are-ignored --all`.
"""

import re

from pathlib import Path

if __name__ == "__main__":
    testing_workflows = ["jaxtests.yml", "pytest.yml"]
    ignored = set()
    non_ignored = set()
    for wfyml in testing_workflows:
        pytest_ci_job = Path(".github") / "workflows" / wfyml
        txt = pytest_ci_job.read_text()
        ignored = set(re.findall(r"(?<=--ignore=)(pymc3/tests.*\.py)", txt))
        non_ignored = non_ignored.union(set(re.findall(r"(?<!--ignore=)(pymc3/tests.*\.py)", txt)))
    assert (
        ignored <= non_ignored
    ), f"The following tests are ignored by the first job but not run by the others: {ignored.difference(non_ignored)}"
    assert (
        ignored >= non_ignored
    ), f"The following tests are run by multiple jobs: {non_ignored.difference(ignored)}"
