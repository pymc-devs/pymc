# pylint: skip-file
import sys


def test_matplotlib_not_imported():
    if 'matplotlib' in sys.modules:
        del sys.modules['matplotlib']

    # make sure matplotlib isn't already imported
    assert 'matplotlib' not in sys.modules

    # make sure matplotlib does not get imported from anything in the top level
    import pymc3 as pm  # noqa
    assert 'matplotlib' not in sys.modules

    # show that it is possible for the test to fail
    import matplotlib  # noqa
    assert 'matplotlib' in sys.modules
