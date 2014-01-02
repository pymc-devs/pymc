try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock

from pymc import progressbar


def test_enumerate_progress():
    iterable = list(range(5, 8))
    meter = mock.Mock()
    results = list(progressbar.enumerate_progress(iterable,
                                                  len(iterable),
                                                  meter))
    for i, _ in enumerate(iterable):
        assert meter.update.called_with(i)
    assert list(zip(*results))[1] == tuple(iterable)
