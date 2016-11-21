import pandas as pd
from pandas.core.common import PandasError
import numpy as np
import theano.tensor as tt


def any_to_tensor_and_labels(x, labels=None):
    """Util for converting input x to tensor trying to
    create labels for columns if they are not provided.

    Default names for columns are ['x0', 'x1', ...], for mappable
    arrays (e.g. pd.DataFrame) their names are treated as labels.
    You can override them with `labels` argument.

    If you have tensor input you should provide labels as we
    cannot get their shape directly

    If you pass dict input we cannot rely on labels order thus dict
    keys are treated as labels anyway

    Parameters
    ----------
    x : np.ndarray | pd.DataFrame | tt.Variable | dict | list
    labels : list - names for columns of output tensor

    Returns
    -------
    (x, labels) - tensor and labels for it's columns
    """
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
        try:
            x = pd.DataFrame.from_dict(x)
            labels = x.columns
            x = x.as_matrix()
        except (PandasError, TypeError):
            res = []
            labels = []
            for k, v in x.items():
                res.append(v)
                labels.append(k)
            x = tt.stack(res, axis=1)
            if x.ndim == 1:
                x = x[:, None]
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
