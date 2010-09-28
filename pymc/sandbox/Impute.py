# =======================================================
# = Implementing imputation as a subclass of Stochastic =
# =======================================================

from pymc.PyMCObjects import Stochastic
import numpy as np
import pdb
class MissingStochastic(Stochastic):
    """Data stochastic with missing values"""

    def __init__(   self,
                    logp,
                    doc,
                    name,
                    parents,
                    random=None,
                    trace=True,
                    value=None,
                    init_value=None,
                    dtype=None,
                    rseed=False,
                    cache_depth=2,
                    plot=None,
                    verbose = None,
                    check_logp=True):
                                
        if not type(value) == np.ma.core.MaskedArray:
        # Generate mask
        
            mask = [v is None or np.isnan(v) for v in value]
            # Generate masked array
            masked_values = np.ma.masked_array(value, mask)
            
        # If there are no intial values given, use mean of available data
        if init_value is None:
            
            init_value = np.resize(masked_values.mean(), sum(masked_values.mask))
            
        # Save data as masked array    
        self._data = masked_values
        
        # Call __init__ of superclass, with missing values as value
        Stochastic.__init__(    self,
                                logp=logp,
                                doc=doc,
                                name=name,
                                parents=parents,
                                random=random,
                                trace=trace,
                                value=init_value,
                                dtype=dtype,
                                rseed=rseed,
                                observed=False,
                                cache_depth=cache_depth,
                                plot=plot,
                                verbose=verbose,
                                check_logp=check_logp)
    
    
    def get_value(self):
        """The current value of the MissingStochastic object, which 
        includes both observed and missing (imputed) values."""
        if self.verbose > 1:
            print '\t' + self.__name__ + ': value accessed.'

        # Embed stochastic values in return value
        return_value = self._data.copy()
        return_value[return_value.mask] = self._value

        return return_value.data.astype(self.dtype)


    def set_value(self, value, force=False):
        # Record new value and increment counter

        if self.verbose > 0:
            print '\t' + self.__name__ + ': value set to ', value

        # Save current value as last_value
        # Don't copy because caching depends on the object's reference.
        self.last_value = self._value

        if self.dtype.kind != 'O':
            self._value = np.asanyarray(value, dtype=self.dtype)
            self._value.flags['W']=False
        else:
            self._value = value

        self.counter.click()

    value = property(fget=get_value, fset=set_value, doc="Self's current value.")
    
    def data():
        doc = "The observed data part of the MissingStochstic object."
        def fget(self):
            if self.verbose > 1:
                print '\t' + self.__name__ + ': observed accessed.'
            return self._data
        return locals()
    data = property(**data())
    
    def missing():
        doc = "The missing data part of the MissingStochastic object"
        def fget(self):
            if self.verbose > 1:
                print '\t' + self.__name__ + ': missing accessed.'
            return self._value
        return locals()
    missing = property(**missing())
    
    get_missing = lambda self: self.missing
        