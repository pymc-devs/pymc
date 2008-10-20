"""
Classes and functions related to data objects, including missing data.
"""

from numpy import array
from Container import ArrayContainer
from distributions import DiscreteUniform, Uniform

__all__ = ['MissingData']

def MissingData(name, iterable, missing=None):
    """
    Container subclass designed for data iterables with missing elements. Elements of the 
    None type (or whatever is specified for missing values) are considered missing values, 
    and are replaced by stochastics. These are given 
    uninformative priors, bounded by the range of the observed data, and are evaluated in the 
    likelihood with the observed data.
    """

    # Array of data with no missing elements
    no_missing = filter(None, iterable)
    # Array type
    dtype = array(no_missing).dtype
    # Use minimum and maximum observed values to bound prior
    minmax = min(no_missing), max(no_missing)
    
    elements = []
    for i,value in enumerate(iterable):
        
        if value is missing:
            
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