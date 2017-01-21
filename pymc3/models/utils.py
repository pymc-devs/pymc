import six
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
    (x, labels) - tensor and labels for its columns
    """
    if isinstance(labels, six.string_types):
        labels = [labels]
    # pandas.DataFrame
    # labels can come from here
    # we can override them
    if isinstance(x, pd.DataFrame):
        if not labels:
            labels = x.columns
        x = x.as_matrix()

    # pandas.Series
    # there can still be a label
    # we can override labels
    elif isinstance(x, pd.Series):
        if not labels:
            labels = [x.name]
        x = x.as_matrix()[:, None]

    # dict
    # labels are keys,
    # cannot override them
    elif isinstance(x, dict):
        # try to do it via pandas
        try:
            x = pd.DataFrame.from_dict(x)
            labels = x.columns
            x = x.as_matrix()
        # some types fail there
        # another approach is to construct
        # variable by hand
        except (PandasError, TypeError):
            res = []
            labels = []
            for k, v in x.items():
                res.append(v)
                labels.append(k)
            x = tt.stack(res, axis=1)
            if x.ndim == 1:
                x = x[:, None]
    # case when it can appear to be some
    # array like value like lists of lists
    # numpy deals with it
    elif not isinstance(x, tt.Variable):
        x = np.asarray(x)
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    # something really strange goes here,
    # but user passes labels trusting seems
    # to be a good option
    elif labels is not None:
        x = tt.as_tensor_variable(x)
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    else:   # trust input
        pass
    # we should check that we can extract labels
    if labels is None and not isinstance(x, tt.Variable):
        labels = ['x%d' % i for i in range(x.shape[1])]
    # for theano variables we should have labels from user
    elif labels is None:
        raise ValueError('Please provide labels as '
                         'we cannot infer shape of input')
    else:   # trust labels, user knows what he is doing
        pass
    # it's time to check shapes if we can
    if not isinstance(x, tt.Variable):
        if not len(labels) == x.shape[1]:
            raise ValueError(
                'Please provide full list '
                'of labels for coefficients, '
                'got len(labels)=%d instead of %d'
                % (len(labels), x.shape[1])
            )
    else:
        # trust labels, as we raised an
        # error in bad case, we have labels
        pass
    # convert labels to list
    if isinstance(labels, pd.RangeIndex):
        labels = ['x%d' % i for i in labels]
    # maybe it was a tuple ot whatever
    elif not isinstance(labels, list):
        labels = list(labels)
    # as output we need tensor
    if not isinstance(x, tt.Variable):
        x = tt.as_tensor_variable(x)
        # finally check dimensions
        if x.ndim == 0:
            raise ValueError('Cannot use scalars')
        elif x.ndim == 1:
            x = x[:, None]
    return x, labels
