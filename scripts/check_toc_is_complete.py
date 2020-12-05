"""
Check that given Jupyter notebooks all appear in the table of contents.

This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-toc --all`.
"""

from pathlib import Path
import argparse
import ast

if __name__ == "__main__":
    toc_examples = (Path("docs") / "source/notebooks/table_of_contents_examples.js").read_text()
    toc_tutorials = (Path("docs") / "source/notebooks/table_of_contents_tutorials.js").read_text()
    toc_keys = {
        **ast.literal_eval(toc_examples[toc_examples.find("{") :]),
        **ast.literal_eval(toc_tutorials[toc_tutorials.find("{") :]),
    }.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    for file_ in args.filenames:
        stem = Path(file_).stem
        assert stem in toc_keys, f"Notebook '{stem}' not added to table of contents!"
