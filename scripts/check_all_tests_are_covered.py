"""
In .github/workflows/pytest.yml, tests are split between multiple jobs.

Here, we check that the jobs ignored by the first job actually end up getting
run by the other jobs.
This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-no-tests-are-ignored --all`.
"""
import itertools
import logging
import os

from pathlib import Path

import pandas
import yaml

_log = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


def find_testfiles():
    dp_repo = Path(__file__).parent.parent
    all_tests = {
        str(fp.relative_to(dp_repo)).replace(os.sep, "/")
        for fp in (dp_repo / "pymc3" / "tests").glob("**/test_*.py")
    }
    _log.info("Found %i tests in total.", len(all_tests))
    return all_tests


def from_yaml():
    """Determins how often each test file is run per platform and floatX setting.

    An exception is raised if tests run multiple times with the same configuration.
    """
    # First collect the matrix definitions from testing workflows
    matrices = {}
    for wf in ["pytest.yml", "arviz_compat.yml"]:
        wfname = wf.strip(".yml")
        wfdef = yaml.safe_load(open(Path(".github", "workflows", wf)))
        for jobname, jobdef in wfdef["jobs"].items():
            matrix = jobdef.get("strategy", {}).get("matrix", {})
            if matrix:
                matrices[(wfname, jobname)] = matrix
            else:
                _log.warning("No matrix in %s/%s", wf, jobname)

    # Now create an empty DataFrame to count based on OS/floatX/testfile
    all_os = []
    all_floatX = []
    for matrix in matrices.values():
        all_os += matrix["os"]
        all_floatX += matrix["floatx"]
    all_os = tuple(sorted(set(all_os)))
    all_floatX = tuple(sorted(set(all_floatX)))
    all_tests = find_testfiles()

    df = pandas.DataFrame(
        columns=pandas.MultiIndex.from_product(
            [sorted(all_floatX), sorted(all_os)], names=["floatX", "os"]
        ),
        index=pandas.Index(sorted(all_tests), name="testfile"),
    )
    df.loc[:, :] = 0

    # Count how often the testfiles are included in job definitions
    for matrix in matrices.values():
        for os_, floatX, subset in itertools.product(
            matrix["os"], matrix["floatx"], matrix["test-subset"]
        ):
            testfiles = subset.split("\n")
            ignored = {item.strip("--ignore=") for item in testfiles if item.startswith("--ignore")}
            included = {item for item in testfiles if item and not item.startswith("--ignore")}
            if ignored and not included:
                # if no testfile is specified explicitly pytest runs all except the ignored ones
                included = all_tests - ignored

            for testfile in included:
                df.loc[testfile, (floatX, os_)] += 1

    ignored_by_all = set(df[df.eq(0).all(axis=1)].index)
    run_multiple_times = set(df[df.gt(1).any(axis=1)].index)

    # Print summary, warnings and raise errors on unwanted configurations
    _log.info("Number of test runs (❌=0, ✅=once)\n%s", df.replace(0, "❌").replace(1, "✅"))

    if ignored_by_all:
        _log.warning("%i tests are completely ignored:\n%s", len(ignored_by_all), ignored_by_all)
    if run_multiple_times:
        raise Exception(
            f"{len(run_multiple_times)} tests are run multiple times with the same OS and floatX setting:\n{run_multiple_times}"
        )
    return


if __name__ == "__main__":
    from_yaml()
