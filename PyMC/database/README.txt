-----------------
Database Backends
-----------------

By default, PyMC keeps the sampled data in memory and keeps no trace of it on the hard drive. To save this data to disk, PyMC provides different strategies, from simple ASCII files to compressed binary formats. These strategies are implemented different *database backends*, behaving identically from the user perspective. In the following, the interface to these backends is discussed, and a description of the different backends is given. 


Accessing Sampled Data: User Interface
=======================================


The choice of database backend is made when a Sampler is created using the `db` keyword::

	S = Sampler(DisasterSampler, db='txt', dirname='test')
	
This instructs the sampler to tally samples in txt files stored in a directory named `test`. Other choices for the database are given in the table below, the default being `ram`. When the `sample` method is called, a `chain` is created storing the sampled variables. The data in this chain can be accessed for each stochastic parameter using its trace object ::

	S.e.trace()

When `S.db.close()` is called, the data is flushed to disk. That is, directories are created for each chain, with samples from each stochastic variable in a separate file. To access this data during a following session, each database provides a `load` function instantiating a `Database` object ::

	DB = Database.txt.load('test')

This object can then be linked to a model definition using ::

	S = Sampler(DisasterSampler, db=DB)

For some databases (`hdf5`, `pickle`), loading an existing database restores the previous state of the sampler. That is, the attribtues of the Sampler, its Stochastic parameters and StepMethods are all set to the value they had at the time `D.db.close()` was called. 


The `trace` object has the following signature .. [#]::

	trace(self,  burn=0, thin=1, chain=-1, slicing=None)

with arguments having the following meaning:

burn : int
	Number of initial samples to skip. 
	
thin : int
	Number of samples to step.

chain : int or None
	Index of the sampling chain to return. Use `None` to return all chains. Note that a new chain is created each time `sample` is called.

slicing : slice
	Slice object used to parse the samples. Overrides burn and thin parameters. 
	
	
.. [#]: The `trace` attribute of stochastic parameters is in fact an instance of a Trace class, defined for each backend. This class has a method called `gettrace` that returns the trace of the object, and which is called by `trace()` . 



Backends description
====================

PyMC provides seven different backends with different level of support. 

ram
---

Used by default, this backend simply holds a copy in memory, with no output written to disk. This is useful for short runs or testing. For long runs generating large amount of data, using this backend may fill the available memory, forcing the OS to store data in the cache, slowing down all running applications on your computer. 

txt
---

The `txt` backend is a modified `ram` backend, the only difference being that when the database is closed, the data is written to disk in ascii files. More precisely, the data for each chain is stored in a directory called `Chain_<#>`, the trace for each variable being stored in a file names`<stoch name>.txt`. This backend makes it easy to load the data using another application, but for large datasets, files tend to be embarassingly large and slow to load into memory. 

pickle
------

As its name implies, the `pickle` database used the `Cpickle` module to save the trace objects. Use of this backend is not suggested since the generated files may become unreadable after a Python update. 

sqlite
------

Chris ...

mysql
-----

Chris ...


hdf5
----

The hdf5 backend uses pyTables to save data in binary HDF5 format. The main advantage of this backend is that data is flushed regularly to disk, reducing memory usage and allowing sampling of datasets much larger than the available memory. Data access is also very fast.




==========  ============================================================  ========================
 Backend     Description                                                   External Dependencies
==========  ============================================================  ========================
 no_trace    Do not tally samples at all. Use only for testing purposes.
 ram         Store samples in memory. 
 txt         Write data to ascii files. 
 pickle      Write data to a pickle file. 
 sqlite      Store samples in a sqlite database.                           sqlite3
 mysql       Store samples in a mysql database.                            MySQLdb
 hdf5        Store samples in the HDF5 format.                             pytables (>2.0), libhdf5
==========  ============================================================  ========================


For more information about individual backends, refer to the `API`_ documentation.

.. _`database/base.py`:
   PyMC/database/base.py

.. _`API`:
   docs/API.pdf
