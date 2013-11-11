import matplotlib
matplotlib.use('Agg', warn=False)

from os import path
import os
import fnmatch
import imp


def test_examples():

    for _path in all_matching_files(example_dir(), '*.py'):
        yield check_example, _path


def all_matching_files(d, pattern):

    def addfiles(fls, dir, nfiles):
        nfiles = fnmatch.filter(nfiles, pattern)
        nfiles = [path.join(dir, f) for f in nfiles]
        fls.extend(nfiles)

    files = []
    os.walk(d, addfiles, files)
    return files


def example_dir():
    import pymc
    d = path.dirname(pymc.__file__)

    return path.join(d, 'examples/')


def check_example(p):
    os.chdir(path.dirname(p))
    imp.load_source('__main__', path.basename(p))
