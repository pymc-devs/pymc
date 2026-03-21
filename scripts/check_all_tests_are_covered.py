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
        for fp in (dp_repo / "tests").glob("**/test_*.py")
    }
    _log.info("Found %i tests in total.", len(all_tests))
    return all_tests


def from_yaml():
    """Determine how often each test file is run per platform and linker setting.

    An exception is raised if tests run multiple times with the same configuration.
    """
    # First collect the matrix definitions from testing workflows
    matrices = {}
    for wf in ["tests.yml"]:
        wfname = wf.rstrip(".yml")
        wfdef = yaml.safe_load(open(Path(".github", "workflows", wf)))
        for jobname, jobdef in wfdef["jobs"].items():
            if jobname in ("float32", "all_tests"):
                continue
            runs_on = jobdef.get("runs-on", "unknown")
            floatX = "float32" if jobname == "float32" else "float64"
            matrix = jobdef.get("strategy", {}).get("matrix", {})
            if matrix:
                # Some jobs are parametrized by os, for others it's fixed
                matrix.setdefault("os", [runs_on])
                matrix.setdefault("floatX", [floatX])
                matrices[(wfname, jobname)] = matrix
            else:
                _log.warning("No matrix in %s/%s", wf, jobname)

    # Now create an empty DataFrame to count based on OS/linker/testfile
    all_os = []
    all_linker = []
    for matrix in matrices.values():
        all_os += matrix["os"]
        all_linker += matrix["linker"]
    all_os = tuple(sorted(set(all_os)))
    all_linker = tuple(sorted(set(all_linker)))
    all_tests = find_testfiles()

    df = pandas.DataFrame(
        columns=pandas.MultiIndex.from_product(
            [sorted(all_linker), sorted(all_os)], names=["linker", "os"]
        ),
        index=pandas.Index(sorted(all_tests), name="testfile"),
    )
    df.loc[:, :] = 0

    # Count how often the testfiles are included in job definitions
    for matrix in matrices.values():
        for os_, linker, subset in itertools.product(
            matrix["os"], matrix["linker"], matrix["test-subset"]
        ):
            lines = [k for k in subset.split("\n") if k]

            # Unpack lines with >1 item
            testfiles = []
            for line in lines:
                testfiles += line.split(" ")

            ignored = {item[8:].lstrip(" =") for item in testfiles if item.startswith("--ignore")}
            included = {item for item in testfiles if item and not item.startswith("--ignore")}

            if ignored and not included:
                # if no testfile is specified explicitly pytest runs all except the ignored ones
                included = all_tests - ignored

            for testfile in included:
                df.loc[testfile, (linker, os_)] += 1

    ignored_by_all = set(df[df.eq(0).all(axis=1)].index)
    run_multiple_times = set(df[df.gt(1).any(axis=1)].index)

    # Print summary, warnings and raise errors on unwanted configurations
    _log.info("Number of test runs (❌=0, ✅=once)\n%s", df.replace(0, "❌").replace(1, "✅"))

    if ignored_by_all:
        raise AssertionError(
            f"{len(ignored_by_all)} tests are completely ignored:\n{ignored_by_all}"
        )
    if run_multiple_times:
        raise AssertionError(
            f"{len(run_multiple_times)} tests are run multiple times with the same OS and pytensor flags:\n{run_multiple_times}"
        )
    return


if __name__ == "__main__":
    from_yaml()
