import matplotlib
matplotlib.use('Agg', warn=False)

from os import path
import os
import fnmatch
import imp


def test_examples():
    for _path in matching_files(example_dir(), '*.py'):
        yield check_example, _path


def matching_files(d, pattern):
    return [path.join(dir, file) 
            for dir, _, files in os.walk(d)
            for file in fnmatch.filter(files, pattern) 
            ]


def example_dir():
    import pymc
    d = path.dirname(pymc.__file__)

    return path.join(d, 'examples/')


def check_example(p):
    os.chdir(path.dirname(p))
    imp.load_source('__main__', path.basename(p))
