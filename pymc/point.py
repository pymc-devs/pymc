import numpy as np


class PointFunc(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(**state)
