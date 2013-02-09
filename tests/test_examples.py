import traceback, sys
from os import path
from glob import glob


def test_examples(): 
    print example_files()
    for path in example_files():
        yield check_example, path


def example_files():
    import pymc
    d = path.dirname(pymc.__file__)

    return glob(path.join(d, "../examples/*.py"))





def check_example(path):
    with open(path) as f :
        c = compile(f.read(), path, 'exec')
        exec(c)
