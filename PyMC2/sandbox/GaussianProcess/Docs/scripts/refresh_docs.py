import getopt, os, sys
from py2tex import Interpret

filenames = os.listdir('../../examples')
files = []
outfiles = {}

for name in filenames:
    if name[-3:] == '.py':
        F = Interpret(  name='../../examples/' + name,
                        docprocess = 'plain')
        
        F.no_math()
        
        out_name = name
        out_name = name[:-1]
        out_name += 't'
        outfile = file(out_name,'w')
        
        while not F.translate() is None:
            for scrap in F.translation():
                outfile.write(scrap)
        outfile.close()
        F.close()


        

