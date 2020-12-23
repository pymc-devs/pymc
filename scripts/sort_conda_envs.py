"""
Sort dependencies in conda environment files.

This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run conda-env-sort --all`.
"""

import argparse

from pathlib import Path
from typing import Optional, Sequence

import ruamel.yaml

yaml = ruamel.yaml.YAML()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Sort dependencies in conda environment files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    for path in args.paths:
        doc = yaml.load(path)
        doc["dependencies"].sort()
        yaml.dump(doc, path)


if __name__ == "__main__":
    main()
