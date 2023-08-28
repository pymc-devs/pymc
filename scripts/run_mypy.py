"""
Invokes mypy and compare the reults with files in /pymc except tests
and a list of files that are known to fail.

Exit code 0 indicates that there are no unexpected results.

Usage
-----
python scripts/run_mypy.py [--verbose]
"""
import argparse
import importlib
import os
import pathlib
import subprocess
import sys

from typing import Iterator

import pandas

DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
FAILING = """
pymc/distributions/continuous.py
pymc/distributions/dist_math.py
pymc/distributions/distribution.py
pymc/distributions/mixture.py
pymc/distributions/multivariate.py
pymc/distributions/timeseries.py
pymc/distributions/truncated.py
pymc/initial_point.py
pymc/logprob/binary.py
pymc/logprob/censoring.py
pymc/logprob/basic.py
pymc/logprob/mixture.py
pymc/logprob/order.py
pymc/logprob/rewriting.py
pymc/logprob/scan.py
pymc/logprob/tensor.py
pymc/logprob/transforms.py
pymc/logprob/utils.py
pymc/model/core.py
pymc/model/fgraph.py
pymc/model/transform/basic.py
pymc/model/transform/conditioning.py
pymc/model_graph.py
pymc/printing.py
pymc/pytensorf.py
pymc/sampling/jax.py
pymc/variational/opvi.py
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


def mypy_to_pandas(input_lines: Iterator[str]) -> pandas.DataFrame:
    """Reformats mypy output with error codes to a DataFrame.

    Adapted from: https://gist.github.com/michaelosthege/24d0703e5f37850c9e5679f69598930a
    """
    current_section = None
    data = {
        "file": [],
        "line": [],
        "type": [],
        "errorcode": [],
        "message": [],
    }
    for line in input_lines:
        line = line.strip()
        elems = line.split(":")
        if len(elems) < 3:
            continue
        try:
            file, lineno, message_type, *_ = elems[0:3]
            message_type = message_type.strip()
            if message_type == "error":
                current_section = line.split("  [")[-1][:-1]
            message = line.replace(f"{file}:{lineno}: {message_type}: ", "").replace(
                f"  [{current_section}]", ""
            )
            data["file"].append(file)
            data["line"].append(lineno)
            data["type"].append(message_type)
            data["errorcode"].append(current_section)
            data["message"].append(message)
        except Exception as ex:
            print(elems)
            print(ex)
    return pandas.DataFrame(data=data).set_index(["file", "line"])


def check_no_unexpected_results(mypy_lines: Iterator[str]):
    """Compares mypy results with list of known FAILING files.

    Exits the process with non-zero exit code upon unexpected results.
    """
    df = mypy_to_pandas(mypy_lines)

    all_files = {
        str(fp).replace(str(DP_ROOT), "").strip(os.sep).replace(os.sep, "/")
        for fp in DP_ROOT.glob("pymc/**/*.py")
        if "tests" not in str(fp)
    }
    failing = set(df.reset_index().file.str.replace(os.sep, "/", regex=False))
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
        print(f"{len(unexpected_failing)} files unexpectedly failed.")
        print("\n".join(sorted(map(str, unexpected_failing))))
        print(
            "These files did not fail before, so please check the above output"
            f" for errors in {unexpected_failing} and fix them."
        )
        print("You can run `python scripts/run_mypy.py --verbose` to reproduce this test locally.")
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
        "--verbose", action="count", default=0, help="Pass this to print mypy output."
    )
    parser.add_argument(
        "--groupby",
        default="file",
        help="How to group verbose output. One of {file|errorcode|message}.",
    )
    args, _ = parser.parse_known_args()

    cp = subprocess.run(
        ["mypy", "--show-error-codes", "--exclude", "tests", "pymc"],
        capture_output=True,
    )
    output = cp.stdout.decode()
    if args.verbose:
        df = mypy_to_pandas(output.split("\n"))
        for section, sdf in df.reset_index().groupby(args.groupby):
            print(f"\n\n[{section}]")
            for row in sdf.itertuples():
                print(f"{row.file}:{row.line}: {row.type}: {row.message}")
        print()
    else:
        print(
            "Mypy output hidden."
            " Run `python run_mypy.py --verbose` to see the full output,"
            " or `python run_mypy.py --help` for other options."
        )

    check_no_unexpected_results(output.split("\n"))
    sys.exit(0)
