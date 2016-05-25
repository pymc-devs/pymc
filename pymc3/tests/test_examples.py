import matplotlib, pkgutil, itertools
from pymc3 import examples 
from nose.plugins.skip import SkipTest

matplotlib.use('Agg', warn=False)

def get_examples():
    prefix = examples.__name__ + "."
    for _, example, _ in pkgutil.iter_modules(examples.__path__):
        yield check_example, prefix + example

        
def check_example(example_name):
    example = __import__(example_name, fromlist='dummy')
    if hasattr(example, 'run'):
        example.run("short")

@SkipTest
def test_examples0():
    for t in itertools.islice(get_examples(), 0, 10):
        yield t
        
@SkipTest
def test_examples1():
    for t in itertools.islice(get_examples(), 10, 20):
        yield t

@SkipTest
def test_examples2():
    for t in itertools.islice(get_examples(), 20, None):
        yield t
