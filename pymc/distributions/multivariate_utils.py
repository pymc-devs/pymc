import numpy as np

from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor import swapaxes
from aesara.tensor.basic import as_tensor_variable


class BatchedMatrixInverse(Op):
    """Computes the inverse of a matrix.

    `aesara.tensor.nlinalg.matrix_inverse` can only inverse square matrices.
    This Op can inverse batches of square matrices.
    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.inv(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        """
        Checkout Page 10 of Matrix Cookbook:
        https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
        """
        (x,) = inputs
        xi = self(x)
        (gz,) = g_outputs

        # Take transpose of last two dimensions
        gz_transpose = swapaxes(gz, -1, -2)

        output = xi * gz_transpose * xi

        output_transpose = swapaxes(output, -1, -2)
        return [-output_transpose]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        (x,) = inputs
        xi = self(x)
        (ev,) = eval_points
        if ev is None:
            return [None]
        return [-xi * ev * xi]

    def infer_shape(self, fgraph, node, shapes):
        return shapes


batched_matrix_inverse = BatchedMatrixInverse()
# import aesara.tensor as at

# array = np.stack([np.eye(3), np.eye(3)])
# array_tensor = at.as_tensor_variable(array)
# at.grad(batched_matrix_inverse(array_tensor).mean(), array_tensor)
