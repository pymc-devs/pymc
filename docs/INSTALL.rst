************
Installation
************

:Date: 1 July 2014
:Authors: Chris Fonnesbeck, Anand Patil, David Huard, John Salvatier
:Contact: chris.fonnesbeck@vanderbilt.edu
:Web site: http://github.com/pymc-devs/pymc
:Copyright: This document has been placed in the public domain.
:License: PyMC is released under the Academic Free license.
:Version: 2.3

PyMC is known to run on Mac OS X, Linux and Windows, but in theory should be
able to work on just about any platform for which Python, a Fortran compiler
and the NumPy module are available. However, installing some extra depencies
can greatly improve PyMC's performance and versatility. The following describes
the required and optional dependencies and takes you through the installation
process.


Dependencies
============

PyMC requires some prerequisite packages to be present on the system.
Fortunately, there are currently only a few hard dependencies, and all are
freely available online.

* `Python`_ version 2.6 or later.

* `NumPy`_ (1.6 or newer): The fundamental scientific programming package, it
  provides a multidimensional array type and many useful functions for
  numerical analysis.

* `Matplotlib`_ (1.0 or newer): 2D plotting library which produces publication
  quality figures in a variety of image formats and interactive environments

* `SciPy`_ (optional): Library of algorithms for mathematics, science and
  engineering.

* `pyTables`_ (optional): Package for managing hierarchical datasets and
  designed to efficiently and easily cope with extremely large amounts of data.
  Requires the `HDF5`_ library.

* `pydot`_ (optional): Python interface to Graphviz's Dot language, it allows
  PyMC to create both directed and non-directed graphical representations of
  models. Requires the `Graphviz`_ library.

* `IPython`_ (optional): An enhanced interactive Python shell and an
  architecture for interactive parallel computing.

* `nose`_ (optional): A test discovery-based unittest extension (required to
  run the test suite).

There are prebuilt distributions that include all required dependencies. For
Mac OS X and Windows users, we recommend the `Anaconda Python distribution`_. Anaconda comes bundled with most of these prerequisites. Note
that depending on the currency of these distributions, some packages may need
to be updated manually.

If, instead of installing the prebuilt binaries, you prefer (or have) to build
``pymc`` yourself, make sure you have a Fortran and a C compiler. There are
free compilers (gfortran, gcc) available on all platforms. Other compilers have
not been tested with PyMC but may work nonetheless.

.. _`Python`: http://www.python.org/.

.. _`NumPy`: http://www.scipy.org/NumPy

.. _`Matplotlib`: http://matplotlib.sourceforge.net/

.. _`SciPy`: http://www.scipy.org/

.. _`IPython`: http://ipython.scipy.org/

.. _`pyTables`: http://www.pytables.org/moin

.. _`HDF5`: http://www.hdfgroup.org/HDF5/

.. _`pydot`: http://code.google.com/p/pydot/

.. _`Graphviz`: http://www.graphviz.org/

.. _`nose`: http://readthedocs.org/docs/nose/en/latest/


Installation using EasyInstall
==============================

The easiest way to install PyMC is to type in a terminal::

  easy_install pymc

Provided `EasyInstall`_ (part of the `setuptools`_ module) is installed and in
your path, this should fetch and install the package from the `Python Package
Index`_. Make sure you have the appropriate administrative privileges to
install software on your computer.

.. _`Python Package Index`: http://pypi.python.org/pypi

.. _`setuptools`: http://peak.telecommunity.com/DevCenter/setuptools


Installing from pre-built binaries
==================================

Pre-built binaries are available for Windows XP and Mac OS X. These can be
installed as follows:

  1. Download the installer for your platform from `PyPI`_ or the `GitHub download page`_.
  2. Double-click the executable installation package, then follow the on-screen
  instructions.

For other platforms, you will need to build the package yourself from source.
Fortunately, this should be relatively straightforward.

.. _`GitHub download page`: http://github.com/pymc-devs/pymc/downloads

Anaconda
--------

If you are running the `Anaconda Python distribution`_ you can install a PyMC binary from the `Binstar`_ package management service, using the `conda` utility:

    conda install -c https://conda.binstar.org/pymc pymc
   
.. _`Anaconda Python distribution`: http://continuum.io/downloads

.. _`Binstar` : https://binstar.org/pymc/pymc

Compiling the source code
=========================

First download the source code from `GitHub`_ and unpack it. Then move
into the unpacked directory and follow the platform specific instructions.

Windows
-------

One way to compile PyMC on Windows is to install `MinGW`_ and `MSYS`_. MinGW is
the GNU Compiler Collection (GCC) augmented with Windows specific headers and
libraries. MSYS is a POSIX-like console (bash) with UNIX command line tools.
Download the `Automated MinGW Installer`_ and double-click on it to launch the
installation process. You will be asked to select which components are to be
installed: make sure the g77 compiler is selected and proceed with the
instructions. Then download and install `MSYS-1.0.exe`_, launch it and again
follow the on-screen instructions.

Once this is done, launch the MSYS console, change into the PyMC directory and
type::

    python setup.py install

This will build the C and Fortran extension and copy the libraries and python
modules in the `site-packages` directory of your Python distribution.

Some Windows users have reported problems building PyMC with MinGW,
particularly under Enthought Python. An alternative approach in this case is
to use the gcc and gfortran compilers that are bundled with EPD (located in the
Scripts directory). In order to do this, you should add the EPD "Scripts"
directory to your PATH environment variable (ensuring that it appears ahead of
the MinGW binary directory, if it exists on your PATH). Then build PyMC using
the install command above.

Alternatively, one may build the currently-available release of PyMC using
`pip`_.


.. _`MinGW`: http://www.mingw.org/

.. _`MSYS`: http://www.mingw.org/wiki/MSYS

.. _`Automated MinGW Installer`: http://sourceforge.net/projects/mingw/files/

.. _`MSYS-1.0.exe`: http://downloads.sourceforge.net/mingw/MSYS-1.0.11.exe

.. _`pip`: http://www.pip-installer.org

Mac OS X or Linux
-----------------

In a terminal, type::

    python setup.py config_fc --fcompiler gfortran build
    python setup.py install

The above syntax also assumes that you have gFortran installed and available.
The `sudo` command may be required to install PyMC into the Python
``site-packages`` directory if it has restricted privileges.

In addition, the python-dev package may be required to install PyMC on Linux systems.


.. _`EasyInstall`: http://peak.telecommunity.com/DevCenter/EasyInstall


.. _`PyPI`: http://pypi.python.org/pypi/pymc/


Installing from GitHub
======================

You can check out the 2.3 branch of PyMC from the `GitHub`_
repository::

    git clone -b 2.3 git://github.com/pymc-devs/pymc.git@2.3

Previous versions are available in the ``/tags`` directory.

.. _`GitHub`: https://github.com/pymc-devs/pymc


Running the test suite
======================

``pymc`` comes with a set of tests that verify that the critical components of
the code work as expected. To run these tests, users must have `nose`_
installed. The tests are launched from a python shell::

    import pymc
    pymc.test()

In case of failures, messages detailing the nature of these failures will
appear. In case this happens (it shouldn't), please report the problems on the
`issue tracker`_ (the issues tab on the GitHub page), specifying the
version you are using and the environment.

.. _`nose`: http://readthedocs.org/docs/nose/en/latest/


Bugs and feature requests
=========================

Report problems with the installation, bugs in the code or feature request at
the `issue tracker`_. Comments and questions are welcome and should be
addressed to PyMC's `mailing list`_.

.. _`issue tracker`: http://github.com/pymc-devs/pymc/issues

.. _`mailing list`: pymc@googlegroups.com
