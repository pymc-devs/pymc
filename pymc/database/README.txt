-----------------
Database Backends
-----------------

In the examples seen so far, traces are simply held in memory and discarded once
the Python session ends. PyMC provides the means to store these traces on disk, 
load them back and add additional samples. Internally, this is implemented in
what we call *database backends*. These backends are simply made of two classes:
``Database`` and ``Trace``, which all present a similar interface to users. 
At the moment, PyMC counts seven such backends: ``ram``, ``no_trace``, 
``pickle``, ``txt``, ``sqlite``, ``mysql`` and ``hdf5``. 
In the following, we present the common interface to those backends and a 
description of each individual backend. 


Accessing Sampled Data
======================

To recommended way to access data from an MCMC run, irrespective of the 
database backend, is to use the ``trace(self, name, chain=-1)`` method::

  >>> M = MCMC(DisasterModel)
  >>> M.sample(10)
  >>> M.trace('e')[:]
  array([ 2.28320992,  2.28320992,  2.28320992,  2.28320992,  2.28320992,
          2.36982455,  2.36982455,  3.1669422 ,  3.1669422 ,  3.14499489])

``M.trace('e')`` returns the ``Trace`` instance associated with the tallyable
object `e`::

  >>> M.trace('e')
  <pymc.database.ram.Trace object at 0x7fa4877a8b50>

This ``Trace`` object from the ``ram`` backend has a ``__getitem__`` method 
that is used to access the trace, just as with any other NumPy array. 
By default, ``trace`` returns the samples from 
the last chain (chain=-1), which in this case is equivalent to ``chain=0``. To 
return the samples from all the chains, use ``chain=None``::

  >>> M.sample(5)
  >>> M.trace('e', chain=None)[:]
  array([ 2.28320992,  2.28320992,  2.28320992,  2.28320992,  2.28320992,
          2.36982455,  2.36982455,  3.1669422 ,  3.1669422 ,  3.14499489,
          3.14499489,  3.14499489,  3.14499489,  2.94672454,  3.10767686])



Saving Data to Disk
===================

By default, the database backend selected by the ``MCMC`` sampler is the ``ram``
backend, which simply holds the data in RAM memory. Now, we will create a 
sampler that, instead, will write data to a pickle file::

  >>> M = MCMC(DisasterModel, db='pickle', dbname='Disaster.pickle')
  >>> M.db
  <pymc.database.pickle.Database object at 0x7fa486623d90>

  >>> M.sample(10)
  >>> M.db.commit()

Note that in this particular case, no data is written to disk before the call
to ``db.commit``. The ``commit`` call creates a file named `Disaster.pickle` 
that contains the trace of each tallyable object as well as the final state of 
the sampler. This means that a user that forgets to call the ``commit`` 
method runs the risk of losing his data. Some backends write the data to disk
continuously, so that not calling ``commit`` is less catastrophic. 

In general, however, it is recommended to always call the ``db.close`` method 
before closing the session. The ``close`` method first calls ``commit``, and 
goes further in making sure that the database is in a safe state. Once ``close``
has been called, further call to ``sample`` will likely fail, at least 
for some backends.  

.. warning::

  Users must absolutly call ``close()`` before closing the session. Otherwise, 
  they run the risk of losing their data. 

Note that all backends except ``ram`` and ``no_trace`` take a ``dbname`` argument that 
specifies the name of the file or directory that will store the data.


Loading Back a Database
=======================

To load the file we just created in a new session, use the ``load`` function
from the backend that created the database::

  >>> import pymc
  >>> db = pymc.database.pickle.load('Disaster.pickle')
  >>> len(db.trace('e')[:])
  10

The ``db`` object also has a ``trace`` method identical to that of ``Sampler``. 
You can hence inspect the results of a model, even when you don't have the model
around. 

To add samples to this file, we need to create an MCMC instance. This time, 
instead of setting ``db='pickle'``, we will pass the existing ``Database``::

  >>> from pymc.examples import DisasterModel
  >>> M = MCMC(DisasterModel, db=db)
  >>> M.sample(5)
  >>> len(M.trace('e', chain=None)[:])
  15
  >>> M.db.close()



Backends Description
====================


ram
---

Used by default, this backend simply holds a copy in memory, with no output 
written to disk. This is useful for short runs or testing. For long runs 
generating large amount of data, using this backend may fill the available 
memory, forcing the OS to store data in the cache, slowing down all running 
applications on your computer. 


no_trace
--------

This backend simply does not store the trace. This may be useful for testing
purposes. 


txt
---

With the ``txt`` backend, the data is written to disk in ASCII files. 
More precisely, the ``dbname`` argument is used to create a top directory 
into which chain directories, called ``Chain_<#>``, are going to be created each 
time ``sample`` is called:: 

    dbname/
      Chain_0/
        <object name>.txt
        <object name>.txt
        ...
      Chain_1/
        <object name>.txt
        <object name>.txt
        ...
      ...

In each one of these chain directories, files named ``<variable name>.txt`` 
are created, storing the values of the variable as rows of text::

  # Variable: e
  # Sample shape: (5,)
  # Date: 2008-11-18 17:19:13.554188
  3.033672373807017486e+00
  3.033672373807017486e+00
  ...

Although this backend makes it easy to load the data using another application, 
for large datasets files tend to be embarassingly large and slow to load 
into memory. 


pickle
------

The ``pickle`` database relies on the ``cPickle`` module to save the 
traces. Use of this backend is appropriate for small scale, 
short-lived projects. For longer term or larger projects, the ``pickle`` 
backend should be avoided since generated files might be unreadable 
across different Python versions. The `pickled` file is a simple dump of a 
dictionary containing the NumPy arrays storing the traces, as well as 
the state of the ``StepMethod``. 



sqlite
------

The ``sqlite`` backend is based on the python module ``sqlite3_`` ( 
a Python 2.5 built-in ) . It opens an SQL database named ``dbname``, 
and creates one table per tallyable objects. The rows of this table
store a key, the chain index and the values of the objects as::

  key (INT), trace (INT),  v1 (FLOAT), v2 (FLOAT), v3 (FLOAT) ... 

The key is autoincremented each time a new row is added to the table. 

.. warning:: Note that the state of the ``StepMethod`` is not saved by 
   the ``sqlite`` backend. 

_`sqlite3` http://www.sqlite.org


mysql
-----

The ``mysql`` backend is based on the ``MySQLdb_`` Python module. The 
database structure is similar to that of the ``sqlite``backend. The 
``mysql.Database`` accepts the following input arguments::




_`MySQLdb`

hdf5
----

The hdf5 backend uses `pyTables`_ to save data in binary HDF5 format. The main advantage of this backend is that data is flushed regularly to disk, reducing memory usage and allowing sampling of datasets much larger than the available RAM memory, speeding up data access. For this backend to work, pyTables must be installed, which in turn requires the hdf5 library. 





==========  =====================================  =========================
 Backend     Description                            External Dependencies
==========  =====================================  =========================
 no_trace    Do not tally samples at all.        
 ram         Store samples in live memory.            
 txt         Write data to ascii files.          
 pickle      Write data to a pickle file.        
 hdf5        Store samples in the HDF5 format.      pytables (>2.0), libhdf5
 sqlite      Store samples in a sqlite database.    sqlite3
 mysql       Store samples in a mysql database.     MySQLdb
==========  =====================================  =========================


For more information about individual backends, refer to the `API`_ documentation.

.. _`database/base.py`:
   pymc/database/base.py

.. _`API`:
   docs/API.pdf

.. _`pyTables`:
   http://www.pytables.org/moin
   
