-----------------
Database backends
-----------------


A typical MCMC run will generate thousands of samples, and some application requires well over 100000 iterations. Keeping all this information in memory can badly strain the performances of PyMC, and users will find their other applications slowing down. Moreover, we generally wish to store all or part of the sampled data for future use. However, there are dozens of different solutions to store data, and each user has his own preference based on previous experience, performance, compatibility, etc. To cover as many user cases as possible, PyMC proposes a database backend. That is, instead of hardcoding data management in the Sampler or Node class, we ask that each parameter is provided with a set of methods taking care of tallying values, and eventually, returning them. PyMC provides a couple of backends for popular data management tools , but users have the possibility to code their own custom made backend, and let Sampler use it seamlessly. 

.. table:: Description of database backends available in PyMC 2.0.

  =========  ==================================================  ================
  Backend    Description                                         Dependencies
  =========  ==================================================  ================
  no_trace   Do not tally samples. Very efficient, mostly used   None
             for testing purposes.
  RAM        Store samples in memory. Efficient for small to     None
             medium size samples. 
  pickle     Store samples in memory, then dump them in a        Cpickle
             pickle file. 
  sqlite     Store samples in a sqlite database.                 sqlite3
  mysql      Store samples in a mysql database.                  MySQLdb
  txt        Store samples in memory, then dump them in a txt    None
             file.
  hdf5       Store samples in the HDF5 format.                   pytables2.0
  =========  ==================================================  ================                     

Backends are selected at Sampler instantiation through the db keyword::
    
	S = Sampler(DisasterSampler, db='sqlite')

Another possibility is to instantiate a Database, then pass it to Sampler::
    
	DB = database.sqlite.Database(filename='test')
	S = Sampler(DisasterSampler, db=DB)

This calling mechanism allows user to pass arguments to the Database, instead 
of relying on the defaults. For databases that provide a load function, it also 
allows user to open an existing database and restart interrupted computations::

	DB = database.pickle.load('results.pickle')
	S = Sampler(DisasterSampler, db=DB)



Use of trace methods
====================

From the user perpective, the only method that really matters is the gettrace()
method (equivalent to trace.__call__). This method returns the values tallied during sampling. So for 
instance::
	
	S = Sampler(DisasterModel, db='ram')
	S.sample(30000,10000,2)

Will tally in memory every other sample, creating arrays of 15000 elements. To 
fetch the last 10000 values of parameter `e` say, we would type::

	S.e.trace(burn=5000)


Backend requirements
====================

Each backend must define minimally two classes: Database and Trace. The Database
class is responsible for opening files, connecting to databases, assigning 
tallyable PyMC objects a Trace instance and calling its methods. Optionnaly, 
the Database class can provide methods to save and return the state of the 
sampler. This is useful for very long computations liable to be stopped then
restarted at a later time. 
The Trace class defines several methods to tally and return the trace of PyMC objects. 

The basic framework of those classes is displayed in `database/base.py`_. Each 
backend subclasses the base clases. 

For more information about individual backends, refer to the `API`_ documentation.

.. _`database/base.py`:
   PyMC2/database/base.py

.. _`API`:
   docs/API.pdf
