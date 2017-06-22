import pytest
import numpy as np
import theano
from theano.configparser import change_flags
from pymc3.variational.updates import (
    sgd,
    momentum,
    nesterov_momentum,
    adagrad,
    rmsprop,
    adadelta,
    adam,
    adamax,
    adagrad_window
)

_a = theano.shared(1.)
_b = _a*2

_m = theano.shared(np.empty((10, ), theano.config.floatX))
_n = _m.sum()
_m2 = theano.shared(np.empty((10, 10, 10), theano.config.floatX))
_n2 = _b + _n + _m2.sum()


@pytest.mark.parametrize(
    'opt',
    [sgd, momentum, nesterov_momentum,
     adagrad, rmsprop, adadelta, adam,
     adamax, adagrad_window],
    ids=['sgd', 'momentum', 'nesterov_momentum',
         'adagrad', 'rmsprop', 'adadelta', 'adam',
         'adamax', 'adagrad_window']
)
@pytest.mark.parametrize(
    'getter',
    [lambda t: t,              # all params -> ok
     lambda t: (None, t[1]),   # missing loss -> fail
     lambda t: (t[0], None),   # missing params -> fail
     lambda t: (None, None)],  # all missing -> partial
    ids=['all_params',
         'missing_loss',
         'missing_params',
         'all_missing']
)
@pytest.mark.parametrize(
    'kwargs',
    [dict(), dict(learning_rate=1e-2)],
    ids=['without_args', 'with_args']
)
@pytest.mark.parametrize(
    'loss_and_params',
    [(_b, [_a]), (_n, [_m]), (_n2, [_a, _m, _m2])],
    ids=['scalar', 'matrix', 'mixed']
)
def test_updates_fast(opt, loss_and_params, kwargs, getter):
    with change_flags(compute_test_value='ignore'):
        loss, param = getter(loss_and_params)
        args = dict()
        args.update(**kwargs)
        args.update(dict(loss_or_grads=loss, params=param))
        if loss is None and param is None:
            updates = opt(**args)
            # Here we should get new callable
            assert callable(updates)
            # And be able to get updates
            updates = opt(_b, [_a])
            assert isinstance(updates, dict)
        # case when both are None is above
        elif loss is None or param is None:
            # Here something goes wrong and user provides not full set of [params + loss_or_grads]
            # We raise Value error
            with pytest.raises(ValueError):
                opt(**args)
        else:
            # Usual call to optimizer, old behaviour
            updates = opt(**args)
            assert isinstance(updates, dict)
