#!/usr/bin/env python
"""
Invoke mypy and compare the reults with files in /pymc.

Excludes tests and a list of files that are known to fail.

Exit code 0 indicates that there are no unexpected results.

Usage
-----
python scripts/run_mypy.py [--verbose]
"""

import argparse
import importlib
import io
import os
import pathlib
import subprocess
import sys

import pandas as pd

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
FAILING = """
pymc/distributions/continuous.py
pymc/distributions/dist_math.py
pymc/distributions/distribution.py
pymc/distributions/custom.py
pymc/distributions/multivariate.py
pymc/distributions/timeseries.py
pymc/distributions/truncated.py
pymc/initial_point.py
pymc/logprob/basic.py
pymc/logprob/mixture.py
pymc/logprob/rewriting.py
pymc/logprob/scan.py
pymc/logprob/transform_value.py
pymc/logprob/transforms.py
pymc/logprob/utils.py
pymc/model/core.py
pymc/model/fgraph.py
pymc/model/transform/conditioning.py
pymc/pytensorf.py
pymc/sampling/jax.py
pymc/sampling/mcmc.py
"""


def enforce_pep561(module_name):
    try:
        module = importlib.import_module(module_name)
        fp = pathlib.Path(module.__path__[0], "py.typed")
        if not fp.exists():
            fp.touch()
    except ModuleNotFoundError:
        print(f"Can't enforce PEP 561 for {module_name} because it is not installed.")
    return


def mypy_to_pandas(mypy_result: str) -> pd.DataFrame:
    """Reformats mypy output with error codes to a DataFrame.

    Adapted from: https://gist.github.com/michaelosthege/24d0703e5f37850c9e5679f69598930a
    """
    return pd.read_json(io.StringIO(mypy_result), lines=True)


def check_no_unexpected_results(mypy_df: pd.DataFrame, show_expected: bool):
    """Compare mypy results with list of known FAILING files.

    Exits the process with non-zero exit code upon unexpected results.
    """
    all_files = {
        str(fp).replace(str(DP_ROOT), "").strip(os.sep).replace(os.sep, "/")
        for fp in DP_ROOT.glob("pymc/**/*.py")
        if "tests" not in str(fp)
    }
    failing = set(mypy_df.file.str.replace(os.sep, "/", regex=False))
    if not failing.issubset(all_files):
        raise Exception(
            "Mypy should have ignored these files:\n"
            + "\n".join(sorted(map(str, failing - all_files)))
        )
    passing = all_files - failing
    expected_failing = set(FAILING.strip().split("\n")) - {""}
    unexpected_failing = failing - expected_failing
    unexpected_passing = passing.intersection(expected_failing)

    if not unexpected_failing:
        print(f"{len(passing)}/{len(all_files)} files pass as expected.")
    else:
        print("!!!!!!!!!")
        print(f"{len(unexpected_failing)} files unexpectedly failed:")
        print("\n".join(sorted(map(str, unexpected_failing))))

        if show_expected:
            print(
                "\nThese files did not fail before, so please check the above output"
                f" for errors in {unexpected_failing} and fix them."
            )
        else:
            print("\nThese files did not fail before. Fix all errors reported in the output above.")
        print("You can run `python scripts/run_mypy.py` to reproduce this test locally.")
        print(
            f"\nNote: In addition to these errors, {len(failing.intersection(expected_failing))} errors in files "
            f'marked as "expected failures" were also found. To see these failures, run: '
            f"`python scripts/run_mypy.py --show-expected`"
        )

        sys.exit(1)

    if unexpected_passing:
        print("!!!!!!!!!")
        print(f"{len(unexpected_passing)} files unexpectedly passed the type checks:")
        print("\n".join(sorted(map(str, unexpected_passing))))
        print(
            "This is good news! Go to scripts/run_mypy.py and remove them from the `FAILING` list."
        )
        if all_files.issubset(passing):
            print("WOW! All files are passing the mypy type checks!")
            print("scripts\\run_mypy.py may no longer be needed.")
        print("!!!!!!!!!")
        sys.exit(1)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mypy type checks on PyMC codebase.")
    parser.add_argument(
        "--verbose", action="count", default=1, help="Pass this to print mypy output."
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Also show expected failures in verbose output.",
    )
    parser.add_argument(
        "--groupby",
        default="file",
        help="How to group verbose output. One of {file|errorcode|message}.",
    )
    args, _ = parser.parse_known_args()

    cp = subprocess.run(
        ["mypy", "--output", "json", "--show-error-codes", "--exclude", "tests", "pymc"],
        stdout=subprocess.PIPE,
    )
    output = cp.stdout.decode("utf-8")
    df = mypy_to_pandas(output)

    if args.verbose:
        if not args.show_expected:
            expected_failing = set(FAILING.strip().split("\n")) - {""}
            df = df.query("file not in @expected_failing")

        for section, sdf in df.groupby(args.groupby):
            print(f"\n\n[{section}]")
            for idx, row in sdf.iterrows():
                print(f"{row.file}:{row.line}: {row.code} [{row.severity}]: {row.message}")
        print()
    else:
        print(
            "Mypy output hidden."
            " Run `python run_mypy.py --verbose` to see the full output,"
            " or `python run_mypy.py --help` for other options."
        )

    check_no_unexpected_results(df, show_expected=args.show_expected)

    sys.exit(0)
