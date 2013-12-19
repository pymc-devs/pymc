import matplotlib
matplotlib.use('Agg', warn=False)

import pkgutil


def test_examples():
    from pymc import examples

    prefix = examples.__name__ + "."
    for _, example, _ in pkgutil.iter_modules(examples.__path__):
        yield check_example, prefix + example

        
def check_example(example_name):
    example = __import__(example_name, fromlist='dummy')
    if hasattr(example, 'run'):
        example.run(n=10)

