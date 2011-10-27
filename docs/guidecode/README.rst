======
README
======

This directory contains code used as examples in the User Guide with
the following modifications:

1. Import statements have been added, since they are assumed valid throughout the
   document::

        import numpy
        import pymc as pm

2. Results from the interpreter are commented with #.

3. >>> have been removed.

4. Code presented as a 'rhetorical' example, not meant to be run, is commented with ##.

5. Some setup variables (eg. indices) are not initialized in the User Guide. In the python code
   present in this directory, the initialization is done and preceded by::

        ## SETUP ##
