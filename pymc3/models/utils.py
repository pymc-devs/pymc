import pandas as pd
import numpy as np
import theano.tensor as tt


def any_to_tensor_and_labels(x, labels=None):
    if isinstance(x, pd.DataFrame):
        labels = list(x.columns)
        x = x.as_matrix()
    elif isinstance(x, pd.Series):
        labels = [x.name]
        x = x.as_matrix()[:, None]
    elif isinstance(x, dict):
        x = pd.DataFrame(x)
        labels = x.columns
        x = x.as_matrix()
    elif not isinstance(x, tt.Variable):
        x = np.asarray(x)
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    elif labels is not None:
        x = tt.as_tensor_variable(x)
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    if labels is None:
        labels = ['x%d' % i for i in range(x.shape[1])]
        # else: labels exist, trust them
    if isinstance(labels, pd.RangeIndex):
        labels = ['x%d' % i for i in labels]
    if not isinstance(labels, list):
        labels = list(labels)
    if not isinstance(x, tt.Variable):
        x = tt.as_tensor_variable(x)
    return x, labels
