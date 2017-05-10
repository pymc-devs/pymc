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
_without_params = dict(loss_or_grads=None, params=None)


@pytest.mark.parametrize(
    ['opt', 'loss_and_params', 'kwargs'],
    [
        [sgd, _with_params, dict()],
        [momentum, _with_params, dict()],
        [nesterov_momentum, _with_params, dict()],
        [adagrad, _with_params, dict()],
        [rmsprop, _with_params, dict()],
        [adadelta, _with_params, dict()],
        [adam, _with_params, dict()],
        [adamax, _with_params, dict()],

        [sgd, _with_params, dict(learning_rate=1e-1)],
        [momentum, _with_params, dict(learning_rate=1e-1)],
        [nesterov_momentum, _with_params, dict(learning_rate=1e-1)],
        [adagrad, _with_params, dict(learning_rate=1e-1)],
        [rmsprop, _with_params, dict(learning_rate=1e-1)],
        [adadelta, _with_params, dict(learning_rate=1e-1)],
        [adam, _with_params, dict(learning_rate=1e-1)],
        [adamax, _with_params, dict(learning_rate=1e-1)],

        [sgd, _without_params, dict()],
        [momentum, _without_params, dict()],
        [nesterov_momentum, _without_params, dict()],
        [adagrad, _without_params, dict()],
        [rmsprop, _without_params, dict()],
        [adadelta, _without_params, dict()],
        [adam, _without_params, dict()],
        [adamax, _without_params, dict()],
    ]
)
@change_flags(compute_test_value='ignore')
def test_updates(opt, loss_and_params, kwargs):
    args = dict(
    )
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
