import matplotlib
matplotlib.use('Agg')

from os import path
import os 
import fnmatch
import imp

def test_examples(): 
    os.chdir(example_dir())

    for path in all_matching_files('.', '*.py'):
        yield check_example, path


def all_matching_files(d, pattern):

    def addfiles(fls, dir, nfiles):
        nfiles = fnmatch.filter(nfiles, pattern)
        nfiles = [path.join(dir,f) for f in nfiles]
        fls.extend(nfiles)
        
    files = []
    path.walk(d, addfiles, files)
    return files

def example_dir():
    import pymc
    d = path.dirname(pymc.__file__)

    return path.join(d, '../examples/')


def check_example(p):
    imp.load_source('__main__', p)

