from ..model import modelcontext
from collections import OrderedDict


class UserModel(object):
    def __init__(self, name):
        # name should be used as prefix for
        # all variables specified within a model
        self.vars = OrderedDict()
        self.name = name

    @property
    def model(self):
        # Just shortcut
        return modelcontext(None)

    def _add_var(self, name, dist, data=None):
        # Signature from pymc3.Model
        var = self.model.Var('{}_{}'.format(self, name, name), dist=dist, data=data)
        self.vars[name] = var
        return var
