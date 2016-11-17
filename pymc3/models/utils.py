import pandas as pd
import numpy as np
import theano.tensor as tt


def any_to_tensor_and_labels(x, labels=None):
    def assert_shape_labels(s, l):
        if not len(l) == s[1]:
            raise ValueError(
                'Please provide full list '
                'of labels for coefficients, '
                'got len(labels)=%d instead of %d'
                % (len(l), s[1])
            )
    if isinstance(x, pd.DataFrame):
        if not labels:
            labels = x.columns
        x = x.as_matrix()
    elif isinstance(x, pd.Series):
        if not labels:
            labels = [x.name]
        x = x.as_matrix()[:, None]
    elif isinstance(x, dict):
        x = pd.DataFrame(x)
        if not labels:
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
    # else: trust input
    if labels is None and not isinstance(x, tt.Variable):
        labels = ['x%d' % i for i in range(x.shape[1])]
    elif labels is None:
        raise ValueError('Please provide labels as '
                         'we cannot infer shape of input')
    # else: trust labels
    if not isinstance(x, tt.Variable):
        assert_shape_labels(x.shape, labels)
    # else: trust labels
    if isinstance(labels, pd.RangeIndex):
        labels = ['x%d' % i for i in labels]
    if not isinstance(labels, list):
        labels = list(labels)
    if not isinstance(x, tt.Variable):
        x = tt.as_tensor_variable(x)
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    return x, labels
