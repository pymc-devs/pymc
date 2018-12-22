import collections

import pytest
from theano import theano

from pymc3.theanof import set_theano_conf


class TestSetTheanoConfig:
    def test_invalid_key(self):
        with pytest.raises(ValueError) as e:
            set_theano_conf({'bad_key': True})
        e.match('Unknown')

    def test_restore_when_bad_key(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            with pytest.raises(ValueError):
                conf = collections.OrderedDict(
                    [('compute_test_value', 'raise'), ('bad_key', True)])
                set_theano_conf(conf)
            assert theano.config.compute_test_value == 'off'

    def test_restore(self):
        with theano.configparser.change_flags(compute_test_value='off'):
            conf = set_theano_conf({'compute_test_value': 'raise'})
            assert conf == {'compute_test_value': 'off'}
            conf = set_theano_conf(conf)
            assert conf == {'compute_test_value': 'raise'}
