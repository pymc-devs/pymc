#!/usr/bin/env python
import sys

for fname in sys.argv[1:]:
    fd = open(fname,mode='U') # open in universal newline mode
    lines = []
    for line in fd.readlines():
        lines.append( line.rstrip() )
    fd.close()

    fd = open(fname,mode='w')
    fd.seek(0)
    for line in lines:
        fd.write(line+'\n')
    fd.close()
