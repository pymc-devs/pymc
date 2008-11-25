from numpy.testing import *
import os

mod_strs = ['IPython', 'pylab', 'matplotlib', 'scipy','Pdb']
class test_dependencies(TestCase):
    def test(self):
        dep_files = {}
        for mod_str in mod_strs:
            dep_files[mod_str] = []

        for dirname, dirs, files in os.walk('..'):
            for fname in files:
                if fname[-3:]=='.py' or fname[-4:]=='.pyx':
                    if dirname.find('sandbox')==-1 and fname != 'test_dependencies.py'\
                        and dirname.find('examples')==-1:
                        for mod_str in mod_strs:
                            if file(dirname+'/'+fname).read().find(mod_str)>=0:
                                dep_files[mod_str].append(dirname+'/'+fname)
                            
        print 'Instances of optional dependencies found are:'
        for mod_str in mod_strs:
            print '\t'+mod_str+':'
            for fname in dep_files[mod_str]:
                print '\t\t'+fname
        if len(dep_files['Pdb'])>0:
            raise ValueError, 'Looks like Pdb was not commented out in '+', '.join(dep_files[mod_str])
                        
            
if __name__ == "__main__":
    import unittest
    unittest.main()