from .compound import *
from .hmc import *
from .metropolis import *
from .gibbs import *
from .slicer import *
from .nuts import *

step_method_registry = [NUTS, HamiltonianMC, Metropolis, BinaryMetropolis, Slice,
                         ElemwiseCategoricalStep]