class StochasticGradient(NormalApproximation):
    """
    At each iteration, randomly chooses one of self.data | self.potentials.
    Updates parameters by gradient descent (with step size parameter) as if
    self's other data and potentials didn't exist.
    """
    def __init__(self):
        raise NotImplemented
