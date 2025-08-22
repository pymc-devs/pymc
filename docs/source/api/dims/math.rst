***************************************
Mathematical operations
***************************************

This module wraps all the mathematical operations defined in :ref:`pytensor.xtensor.math <pytensor:libdoc_xtensor_math>`.

It includes a ``linalg`` submodule that wraps all the operations defined in :ref:`pytensor.xtensor.linalg <pytensor:libdoc_xtensor_linalg>`.

Operations defined at the module level in :ref:`pytensor.xtensor <pytensor:libdoc_xtensor_module_function>` are available at the :mod:`pymc.dims` module level.

Method-based API on XTensorVariable
===================================

Many operations are also available as methods on :class:`~pytensor.xtensor.type.XTensorVariable` instances, providing a more intuitive interface when working with named dimensions:

**Key methods include:**

- :meth:`~pytensor.xtensor.type.XTensorVariable.isel` - Select data by dimension names
- :meth:`~pytensor.xtensor.type.XTensorVariable.dot` - Perform dot products
- :meth:`~pytensor.xtensor.type.XTensorVariable.sum`, :meth:`~pytensor.xtensor.type.XTensorVariable.mean`, :meth:`~pytensor.xtensor.type.XTensorVariable.prod` - Aggregation operations
- :meth:`~pytensor.xtensor.type.XTensorVariable.transpose` - Reorder dimensions by name
- :meth:`~pytensor.xtensor.type.XTensorVariable.expand_dims`, :meth:`~pytensor.xtensor.type.XTensorVariable.squeeze` - Dimension manipulation

Example
-------

.. code-block:: python

    import numpy as np
    import pymc as pm
    
    coords = {"time": np.arange(10), "feature": ["a", "b", "c"]}
    with pm.Model(coords=coords):
        x = pm.Normal("x", mu=0, sigma=1, dims=("time", "feature"))
        
        # Method-based selection and operations
        x_subset = x.isel(time=slice(0, 5))  # First 5 timesteps
        x_mean = x_subset.mean(dim="feature")  # Average across features
        
        # Dot product using method syntax
        y = pm.Normal("y", mu=0, sigma=1, dims="feature")
        dot_result = x.dot(y, dim="feature")  # Shape: (time,)

For a complete list of available methods, see the :class:`~pytensor.xtensor.type.XTensorVariable` API documentation.
