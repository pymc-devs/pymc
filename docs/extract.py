#!/usr/bin/python

import re
from optparse import OptionParser
import sys

def fextract(text, start=None, end=None):
    """Return the text between regular expressions start and end."""
    if type(text) is list:
        text = ''.join(text)
    if start is not None:
        text = re.split(start, text)[1]
    if end is not None:
        text =  re.split(end, text)[0]
    return text
    
parser=OptionParser()
parser.add_option('-f', '--file', dest='filename', help='Input data file', default=None)
parser.add_option('-o', '--outputfile', dest='output', help='Output data file', default=None)
parser.add_option('-s', '--start', dest='start', help='Start regexp', default=None)
parser.add_option('-e', '--end', dest='end', help='End regexp', default=None)
parser.usage = 'extract.py -s start -e end [options]'
options, args = parser.parse_args(sys.argv[1:])
#print options, args

##if len(args) ==0:
##    parser.print_usage()
##    sys.exit()
##else:
##    start = args[0]
##    end = args[0]
    
if options.filename is not None:
    text = open(options.filename,'r').readlines()
else:
    text=sys.stdin.readlines()
       
if options.output is None:
    output = sys.stdout
else:
    output = open(options.output, 'w')
    
out = fextract(text, options.start, options.end)
output.writelines(out)
