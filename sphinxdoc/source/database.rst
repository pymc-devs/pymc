~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Saving and managing sampling results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, PyMC keeps the sampled data in memory and keeps no trace of it on the hard drive. To save this data to disk, PyMC provides different storing strategies, which we refer to as *database backends*. All those backends provide the same user interface, making it trivial to switch from one backend to another. In the following, this common interface is presented, along with an individual description of each backend. 

Accessing Sampled Data: User Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The database backend is selected by the `db` keyword::

	S = MCMC(DisasterModel, db='ram')
	
Here, we instructed the MCMC sampler to keep the trace in the computer's live memory. This means that when the Python session closes, all data will be lost. This is the default backend. 

Each time MCMC's `sample` method is called, a `chain` is created storing the sampled variables. The data in this chain can be accessed for each variable using the call method of its trace attribute::

	S.e.trace(burn=0, thin=1, chain=-1, slicing=None)

with arguments having the following meaning:

burn : int
	Number of initial samples to skip. 
	
thin : int
	The stride, ie the number of samples to step for each returned value.

chain : int or None
	Index of the sampling chain to return. Use `None` to return all chains. Note that a new chain is created each time `sample` is called.

slicing : slice
	Slice object used to parse the samples. Overrides burn and thin parameters. 
	



Loading data from a previous session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To store a copy of the trace on the hard disk, a number of backends are available: `txt`, `pickle`, `hdf5`, `sqlite` and `mysql`. These all write the data to disk, in such a way that it can be loaded back in a following session and appended to. So for instance, to save data in ASCII format, we would do::
 
	S = MCMC(DisasterModel, db='txt', dirname='disaster_data')
	S.sample(10000)
	S.db.close()

When `S.db.close()` is called, the data is flushed to disk. That is, directories are created for each chain, with samples from each stochastic variable in a separate file. To access this data during a following session, each database provides a `load` function instantiating a `Database` object ::

	DB = Database.txt.load('disaster_data')

This `Database` object can then be linked to a model definition using ::

	S = Sampler(DisasterSampler, db=DB)
	S.sample(10000)

For some databases (`hdf5`, `pickle`), loading an existing database restores the previous state of the sampler. That is, the attributes of the Sampler, its Stochastic parameters and StepMethods are all set to the value they had at the time `S.db.close()` was called. 



Backends description
~~~~~~~~~~~~~~~~~~~~

PyMC provides seven different backends with different level of support. 


**ram**
  Used by default, this backend simply holds a copy in memory, with no output written to disk. This is useful for short runs or testing. For long runs generating large amount of data, using this backend may fill the available memory, forcing the OS to store data in the cache, slowing down all running applications on your computer. 

**txt**
  With the `txt` backend, the data is written to disk in ASCII files when the class `close()` method is called. More precisely, the data for each chain is stored in a directory called `Chain_<#>`, the trace for each variable being stored in a file names`<variable name>.txt`. This backend makes it easy to load the data using another application, but for large datasets, files tend to be embarassingly large and slow to load into memory. 

**pickle**
  As its name implies, the `pickle` database relies on the `Cpickle` module to save the trace objects. Use of this backend is appropriate for small scale, short-lived projects. For longer term or larger projects, the `pickle` backend should be avoided since generated files might be unreadable across different Python versions. 

**hdf5**
  The hdf5 backend uses `pyTables`_ to save data in binary HDF5 format. The main advantage of this backend is that data is flushed regularly to disk, reducing memory usage and allowing sampling of datasets much larger than the available RAM memory, speeding up data access. For this backend to work, pyTables must be installed, which in turn requires the hdf5 library. 

**sqlite**
  The sqlite backend is based on the python module sqlite3. It is not as mature as the other backends, in the sense that is does not support saving/restoring of state and plug and play reloading.

**mysql**
  The mysql backend is based on the MySQLdb python module. It also is not as mature as the other backends. 


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
   
