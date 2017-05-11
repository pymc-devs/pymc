import pytest
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
    adamax
)

_a = theano.shared(1.)
_b = _a*2

_with_params = dict(loss_or_grads=_b, params=[_a])
_without_first_param = dict(loss_or_grads=None, params=[_a])
_without_second_param = dict(loss_or_grads=_b, params=None)
_without_params = dict(loss_or_grads=None, params=None)


@pytest.mark.parametrize(
    'opt',
    [sgd, momentum, nesterov_momentum, adagrad, rmsprop, adadelta, adam, adamax]
)
@pytest.mark.parametrize(
    'loss_and_params',
    [_with_params, _without_first_param, _without_second_param, _without_params]
)
@pytest.mark.parametrize(
    'kwargs',
    [dict(), dict(learning_rate=1e-2)]
)
def test_updates_fast(opt, loss_and_params, kwargs):
    with change_flags(compute_test_value='ignore'):
        args = dict()
        args.update(**kwargs)
        args.update(**loss_and_params)
        if loss_and_params['loss_or_grads'] is None and loss_and_params['params'] is None:
            updates = opt(**args)
            # Here we should get new callable
            assert callable(updates)
            # And be able to get updates
            updates = opt(_b, [_a])
            assert isinstance(updates, dict)
        elif loss_and_params['loss_or_grads'] is None or loss_and_params['params'] is None:
            # Here something goes wrong and user provides not full set of [params + loss_or_grads]
            # We raise Value error
            with pytest.raises(ValueError):
                opt(**args)
        else:
            # Usual call to optimizer, old behaviour
            updates = opt(**args)
            assert isinstance(updates, dict)
