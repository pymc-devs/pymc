from .advi import advi, sample_vp
from .advi_minibatch import advi_minibatch

from .updates import (sgd,
                      apply_momentum,
                      momentum,
                      apply_nesterov_momentum,
                      nesterov_momentum,
                      adagrad,
                      rmsprop,
                      adadelta,
                      adam,
                      adamax,
                      norm_constraint,
                      total_norm_constraint
)

from .svgd import svgd
