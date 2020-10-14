"""
Check that given Jupyter notebooks all contain a final watermark cell to facilite reproducibility.

This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run watermark --all`.
"""

import argparse
from pathlib import Path
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    for file_ in args.filenames:
        assert (
            re.search(
                r"%load_ext watermark.*%watermark -n -u -v -iv -w",
                Path(file_).read_text(),
                flags=re.DOTALL,
            )
            is not None
        ), (
            f"Watermark not found in {file_} - please see the PyMC3 Jupyter Notebook Style guide:\n"
            "https://github.com/pymc-devs/pymc3/wiki/PyMC's-Jupyter-Notebook-Style"
        )
