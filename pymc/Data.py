"""
Classes and functions related to data objects, including missing data.
"""

from numpy import array
from Container import ArrayContainer
from distributions import DiscreteUniform, Uniform

__all__ = ['MissingData']

def MissingData(name, iterable, missing=None):
    """
    Returns container designed for data iterables with missing elements. Elements of the 
    None type (or whatever is specified for missing values) are considered missing values, 
    and are replaced by stochastics. These are given uninformative priors, bounded by the 
    range of the observed data, and are evaluated in the likelihood with the observed data.
    
    :Parameters:
        - name (string) : name for the data
        - iterable (iterable) : the data which contains missing values
        - missing (optional) : the value that is considered missing e.g., None (default), -999
    
    :Example:
    
        >>> some_data = (3, -999, 1, 5, 4, 1, 2)
        >>> x = MissingData('x', some_data, -999)
        >>> x
        ArrayContainer([3, x_1, 1, 5, 4, 1, 2], dtype=object)
        >>> x.value
        array([3, 5, 1, 5, 4, 1, 2], dtype=object)
        
    """

    # Array of data with no missing elements
    no_missing = filter(lambda x: x!=missing, iterable)
    # Array type
    dtype = array(no_missing).dtype
    # Use minimum and maximum observed values to bound prior
    minmax = min(no_missing), max(no_missing)
    
    elements = []
    for i,value in enumerate(iterable):
        
        if value == missing:
            
            # Give missing element a name
            missing_name = '%s_%i' % (name, i)
        
            # Uninformative priors according to type
            if dtype==int:
                elements.append(DiscreteUniform(missing_name, minmax[0], minmax[1]))
            else:
                elements.append(Uniform(missing_name, minmax[0], minmax[1]))
                
        else:
            elements.append(value)
    
    return ArrayContainer(array(elements))