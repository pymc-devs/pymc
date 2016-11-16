import pandas as pd
import theano
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
    else:
        if hasattr(x, 'shape') and not isinstance(x, tt.Variable):
            shape = x.shape
        else:
            shape = None
        _old_x_ = x
        x = tt.as_tensor_variable(x)
        if shape is None:
            try:
                shape = x.shape.eval()
            except theano.gof.fg.MissingInputError:
                pass
        if not labels and shape is None:
            raise TypeError(
                'Cannot infer shape for %r, '
                'please provide list of labels '
                'or `x` without missing inputs' % _old_x_
            )
        if shape is not None:
            if len(shape) == 0:
                raise ValueError('scalars are not available as input')
            elif len(shape) == 1:
                x = x[:, None]
                shape = x.shape.eval()
            if not labels:
                labels = ['x%d' % i for i in range(shape[1])]
            else:
                assert len(labels) == shape[1], (
                    'Please provide all labels for coefficients'
                )
        # else: labels exist
    if isinstance(labels, pd.RangeIndex):
        labels = ['x%d' % i for i in labels]
    if not isinstance(labels, list):
        labels = list(labels)
    x = tt.as_tensor_variable(x)
    return x, labels
