'''
Created on Oct 24, 2009

# Author: John Salvatier <jsalvati@u.washington.edu>, 2009.
'''
from distutils.core import setup
from distutils.extension import Extension


DISTNAME            = 'mcex'
DESCRIPTION         = "Monte-Carlo Experiment"
LONG_DESCRIPTION    =""""""
MAINTAINER          = 'John Salvatier'
MAINTAINER_EMAIL    = "jsalvati@u.washington.edu"
URL                 = ""
LICENSE             = "BSD"
VERSION             = "0.1"

classifiers =  ['Development Status :: 3 - Alpha',
                'Programming Language :: Python',
                'License :: OSI Approved :: BSD License',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Operating System :: OS Independent']

if __name__ == "__main__":

    setup(name = DISTNAME,
          version = VERSION,
        maintainer  = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        description = DESCRIPTION,
        license = LICENSE,
        url = URL,
        long_description = LONG_DESCRIPTION,
        packages = ['mcex'], 
        classifiers =classifiers,
        install_requires=['theano','numpy','scipy','numdifftools'])

