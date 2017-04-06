import glob
import os
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'docs', 'source', 'notebooks')


def list_notebooks():
    """Get an iterator of filepaths to notebooks in NOTEBOOK_DIR"""
    return glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb'))


def execute_notebook(notebook_path):
    """Run and overwrite a notebook file."""
    ep = ExecutePreprocessor(timeout=-1)
    with open(notebook_path, 'r') as buff:
        nb = nbformat.read(buff, as_version=nbformat.NO_CONVERT)
    try:
        t0 = time.time()
        ep.preprocess(nb, {'metadata': {'path': NOTEBOOK_DIR}})
        t1 = time.time()

    except KeyboardInterrupt:
        raise

    except BaseException as e:
        t1 = time.time()
        return False, 'Failed after {:.1f}s:\n{}'.format(t1 - t0, str(e))

    with open(notebook_path, 'w') as buff:
        nbformat.write(nb, buff)

    return True, 'Succeeded after {:.1f}s'.format(t1 - t0)


def run_all_notebooks():
    """Try to re-run all notebooks.  Print failures at end."""
    failed = []
    for notebook_path in list_notebooks():
        print('Executing {}'.format(os.path.basename(notebook_path)))
        success, message = execute_notebook(notebook_path)
        if not success:
            failed.append(os.path.basename(notebook_path))
        print(message)

    if failed:
        print('The following notebooks had errors!')
        print('\n'.join(failed))


if __name__ == '__main__':
    run_all_notebooks()
