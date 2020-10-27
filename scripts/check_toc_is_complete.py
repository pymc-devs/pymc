"""
Check that given Jupyter notebooks all contain a final watermark cell to facilite reproducibility.

This is intended to be used as a pre-commit hook, see `.pre-commit-config.yaml`.
You can run it manually with `pre-commit run check-toc --all`.
"""

import json
from pathlib import Path

if __name__ == "__main__":
    notebooks = (Path("docs") / "source/notebooks").glob("*.ipynb")
    toc_examples = (Path("docs") / "source/notebooks/table_of_contents_examples.js").read_text()
    toc_tutorials = (Path("docs") / "source/notebooks/table_of_contents_tutorials.js").read_text()
    toc_keys = {
        **json.loads(toc_examples[toc_examples.find("{") :]),
        **json.loads(toc_tutorials[toc_tutorials.find("{") :]),
    }.keys()
    for notebook in notebooks:
        assert (
            notebook.stem in toc_keys
        ), f"Notebook {notebook.name} not added to table of contents!"
