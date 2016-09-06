import functools
import nose


def knownfailure(msg):
    def decorator(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            try:
                test(*args, **kwargs)
            except Exception:
                raise nose.SkipTest
            else:
                raise AssertionError('Failure expected: ' + msg)
        return inner
    return decorator
