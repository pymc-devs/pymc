# pylint: skip-file
import sys

from pymc3 import *  # noqa


def test_matplotlib_not_imported():
    assert 'matplotlib' not in sys.modules.keys()
