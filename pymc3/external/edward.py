"""
Copied and modified from Edward code base (https://github.com/blei-lab/edward) following
deprecation of external package wrappers by Edward.  Requires installation of tensorflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf  # pragma pylint: disable=import-error

class EdwardModel(object):
    """Model wrapper for models written in PyMC3 to be fit using Edward.
    """
    def __init__(self, model):
        """
        Parameters
        ----------
        model : pymc3.Model
            The probability model, written with Theano shared
            variables to form any observations and with
            `transform=None` for any latent variables. The Theano
            shared variables are set during inference, and all latent
            variables live on their original (constrained) space.
        """

        self.model = model
        self.n_vars = None

    def log_prob(self, xs, zs):
        """
        Parameters
        ----------
        xs : dict of str to tf.Tensor
            Data dictionary. Each key is a data structure used in the
            model (Theano shared variable), and its value is the
            corresponding realization (tf.Tensor).
        zs : dict of str to tf.Tensor
            Latent variable dictionary. Each key names a latent variable
            used in the model (str), and its value is the corresponding
            realization (tf.Tensor).

        Returns
        -------
        tf.Tensor
            Scalar, the log joint density log p(xs, zs).

        Notes
        -----
        It wraps around a Python function. The Python function takes
        inputs of type np.ndarray and outputs a np.ndarray.
        """
        # Store keys so that ``_py_log_prob_args`` knows how each
        # value corresponds to a key.
        self.xs_keys = list(six.iterkeys(xs))
        self.zs_keys = list(six.iterkeys(zs))

        # Pass in all tensors as a flattened list for tf.py_func().
        inputs = [tf.convert_to_tensor(x) for x in six.itervalues(xs)]
        inputs += [tf.convert_to_tensor(z) for z in six.itervalues(zs)]

        return tf.py_func(self._py_log_prob_args, inputs, [tf.float32])[0]

    def _py_log_prob_args(self, *args):
        xs_values = args[:len(self.xs_keys)]
        zs_values = args[len(self.xs_keys):]

        # Set data placeholders in PyMC3 model (Theano shared
        # variable) to their realizations (NumPy array).
        for key, value in zip(self.xs_keys, xs_values):
            key.set_value(value)

        # Calculate model's log density using a dictionary of latent
        # variables.
        z = {key: value for key, value in zip(self.zs_keys, zs_values)}
        lp = self.model.fastlogp(z)
        return lp.astype(np.float32)
