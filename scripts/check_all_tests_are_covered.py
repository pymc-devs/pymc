"""
In .github/workflows/pytest.yml, tests are split between multiple jobs.

Here, we check that the jobs ignored by the first job actually end up getting
run by the other jobs.
This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-no-tests-are-ignored --all`.
"""
import logging
import re

from pathlib import Path

_log = logging.getLogger(__file__)


if __name__ == "__main__":
    testing_workflows = ["jaxtests.yml", "pytest.yml"]
    ignored = set()
    non_ignored = set()
    for wfyml in testing_workflows:
        pytest_ci_job = Path(".github") / "workflows" / wfyml
        txt = pytest_ci_job.read_text()
        ignored = set(re.findall(r"(?<=--ignore=)(pymc3/tests.*\.py)", txt))
        non_ignored = non_ignored.union(set(re.findall(r"(?<!--ignore=)(pymc3/tests.*\.py)", txt)))
    # Summarize
    ignored_by_all = ignored.difference(non_ignored)
    run_multiple_times = non_ignored.difference(ignored)

    if ignored_by_all:
        _log.warning(
            f"The following {len(ignored_by_all)} tests are completely ignored: {ignored_by_all}"
        )
    if run_multiple_times:
        _log.warning(
            f"The following {len(run_multiple_times)} tests are run multiple times: {run_multiple_times}"
        )
    if not (ignored_by_all or run_multiple_times):
        print(f"✔ All tests will run exactly once.")

    # Temporarily disabled as we're bringing features back for v4:
    # assert not ignored_by_all
    assert not run_multiple_times
