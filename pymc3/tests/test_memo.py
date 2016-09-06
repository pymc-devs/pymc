from pymc3.memoize import memoize


def getmemo():
    @memoize
    def f(a, b=['a']):
        return str(a) + str(b)
    return f


def test_memo():
    f = getmemo()

    assert f('x', ['y', 'z']) == "x['y', 'z']"
    assert f('x', ['a', 'z']) == "x['a', 'z']"
    assert f('x', ['y', 'z']) == "x['y', 'z']"
