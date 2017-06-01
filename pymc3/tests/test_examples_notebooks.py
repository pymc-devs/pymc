import os
from glob import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

from .helpers import SeededTest

notebooks_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'source', 'notebooks')
notebooks_files = [file for file in os.listdir(notebooks_dir) if file.lower().endswith('.ipynb')]

class TestExamplesNotebooks(SeededTest):
    @pytest.mark.parametrize('notebook_filename', notebooks_files)
    def test_run_notebook(self, notebook_filename):
        with open(os.path.join(notebooks_dir, notebook_filename)) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=300)
        ep.preprocess(nb, {'metadata': {'path': notebooks_dir}})
