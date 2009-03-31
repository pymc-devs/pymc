*************************
Installation Instructions
*************************

:Date: Oct 14, 2008
:Authors: Chris Fonnesbeck, Anand Patil, David Huard
:Web site: http://code.google.com/p/pymc/
:Copyright: This document has been placed in the public domain.

PyMC is known to run on Mac OS X, Linux and Windows, but in theory should be
able to work on just about any platform for which Python, a Fortran compiler
and the NumPy module are  available. However, installing some extra
depencies can greatly improve PyMC's performance and versatility.
The following describes the required and optional dependencies and takes you
through the installation process.



Dependencies
============

PyMC requires some prerequisite packages to be present on the system.
Fortunately, there are currently only a few dependencies, and all are
freely available online.

* `Python`_ version 2.5.

* `NumPy`_ (1.2 or newer): The fundamental scientific programming package, it provides a
  multidimensional array type and many useful functions for numerical analysis.

* `Matplotlib`_ (optional): 2D plotting library which produces publication
  quality figures in a variety of image formats and interactive environments

* `pyTables`_ (optional): Package for managing hierarchical datasets and
  designed to efficiently and easily cope with extremely large amounts of data.
  Requires the `HDF5`_ library.

* `pydot`_ (optional): Python interface to Graphviz's Dot language, it allows
  PyMC to create both directed and non-directed graphical representations of models.
  Requires the `Graphviz`_ library.

* `SciPy`_ (optional): Library of algorithms for mathematics, science
  and engineering.

* `IPython`_ (optional): An enhanced interactive Python shell and an
  architecture for interactive parallel computing.

* `nose`_ (optional): A test discovery-based unittest extension (required
  to run the test suite).


There are prebuilt distributions that include all required dependencies. For
Mac OSX users, we recommend the `MacPython`_ distribution, the
`Enthought Python Distribution`_, or Python 2.5.1 that ships with
OSX 10.5 (Leopard). Windows users should download and install the
`Enthought Python Distribution`_. The Enthought Python Distribution comes
bundled with these prerequisites. Note that depending on the currency of these
distributions, some packages may need to be updated manually.

If instead of installing the prebuilt binaries you prefer (or have) to build
``pymc`` yourself, make sure you have a Fortran and a C compiler. There are free
compilers (gfortran, gcc) available on all platforms. Other compilers have not been
tested with PyMC but may work nonetheless.


.. _`Python`: http://www.python.org/.

.. _`NumPy`: http://www.scipy.org/NumPy

.. _`Matplotlib`: http://matplotlib.sourceforge.net/

.. _`MacPython`: http://www.activestate.com/Products/ActivePython/

.. _`Enthought Python Distribution`:
   http://www.enthought.com/products/epddownload.php

.. _`SciPy`: http://www.scipy.org/

.. _`IPython`: http://ipython.scipy.org/

.. _`pyTables`: http://www.pytables.org/moin

.. _`HDF5`: http://www.hdfgroup.org/HDF5/

.. _`pydot`: http://code.google.com/p/pydot/

.. _`Graphviz`: http://www.graphviz.org/


Installation using EasyInstall
==============================

The easiest way to install PyMC is to type in a terminal::

  easy_install pymc

Provided `EasyInstall`_ (part of the `setuptools`_ module) is installed
and in your path, this should fetch and install the package from the
`Python Package Index`_. Make sure you have the appropriate administrative
privileges to install software on your computer.

.. _`Python Package Index`: http://pypi.python.org/pypi


.. _`setuptools`: http://peak.telecommunity.com/DevCenter/setuptools


Installing from pre-built binaries
==================================

Pre-built binaries are available for Windows XP and Mac OS X. There are at least
two ways to install these:

1. Download the installer for your platform from `PyPI`_.

2. Double-click the executable installation package, then follow the
   on-screen instructions.

For other platforms, you will need to build the package yourself from source.
Fortunately, this should be relatively straightforward.

.. _`PyMC site`: pymc.googlecode.com


Compiling the source code
=========================

First download the source code tarball from `PyPI`_ and unpack it. Then move
into the unpacked directory and follow the platform specific instructions.

Windows
-------

One way to compile PyMC on Windows is to install `MinGW`_ and `MSYS`_. MinGW is
the GNU Compiler Collection (GCC) augmented with Windows specific headers and
libraries. MSYS is a POSIX-like console (bash) with UNIX command line tools.
Download the `Automated MinGW Installer`_ and double-click on it to launch
the installation process. You will be asked to select which
components are to be installed: make sure the g77 compiler is selected and
proceed with the instructions. Then download and install `MSYS-1.0.exe`_,
launch it and again follow the on-screen instructions.

Once this is done, launch the MSYS console, change into the PyMC directory and
type::

    python setup.py install

This will build the C and Fortran extension and copy the libraries and python
modules in the C:/Python25/Lib/site-packages/pymc directory.


.. _`MinGW`: http://www.mingw.org/

.. _`MSYS`: http://www.mingw.org/wiki/MSYS

.. _`Automated MinGW Installer`: http://sourceforge.net/project/showfiles.php?group_id=2435

.. _`MSYS-1.0.exe`: http://downloads.sourceforge.net/mingw/MSYS-1.0.10.exe


Mac OS X or Linux
-----------------
In a terminal, type::

    python setup.py build
    sudo python setup.py install

The `sudo` command is required to install PyMC into the Python ``site-packages``
directory if it has restricted privileges. You will be prompted for a password,
and provided you have superuser privileges, the installation will proceed.


.. _`EasyInstall`: http://peak.telecommunity.com/DevCenter/EasyInstall


.. _`PyPI`: http://pypi.python.org/pypi/pymc/


Development version
===================

You can check out the bleeding edge version of the code from the `subversion`_
repository::

    svn checkout http://pymc.googlecode.com/svn/trunk/ pymc

Previous versions are available in the ``/tags`` directory.

.. _`subversion`: http://subversion.tigris.org/


Running the test suite
======================

``pymc`` comes with a set of tests that verify that the critical components
of the code work as expected. To run these tests, users must have `nose`_
installed. The tests are launched from a python shell::

    import pymc
    pymc.test()

In case of failures, messages detailing the nature of these failures will
appear. In case this happens (it shouldn't), please report
the problems on the `issue tracker`_, specifying the version you are using and
the environment.

.. _`nose`: http://somethingaboutorange.com/mrl/projects/nose/




Bugs and feature requests
=========================

Report problems with the installation, bugs in the code or feature request at
the `issue tracker`_. Comments and questions are welcome and should be
addressed to PyMC's `mailing list`_.


.. _`issue tracker`: http://code.google.com/p/pymc/issues/list .

.. _`mailing list`: pymc@googlegroups.com
