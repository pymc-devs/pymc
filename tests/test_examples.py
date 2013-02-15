import matplotlib
matplotlib.use('Agg')

import os
from os import path
from glob import glob
import fnmatch

def test_examples(): 
    for path in all_matching_files(example_dir(), '*.py'):
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
    with open(p) as f :
        #c = compile(f.read(), p, 'exec')
        #exec(c)
        os.chdir(path.dirname(p))
        execfile(path.basename(p))
