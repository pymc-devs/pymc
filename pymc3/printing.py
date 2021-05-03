#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from aesara.graph.basic import walk
from aesara.tensor.basic import TensorVariable
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import RandomVariable
from aesara.tensor.var import TensorConstant


class PrettyPrintingTensorVariable(TensorVariable):
    def __str__(self):
        print_name = self.name if self.name is not None else "<unnamed>"
        dist_name = self.owner.op._print_name[0]

        # first 3 args are always (rng, size, dtype), rest is relevant for distribution
        dist_args = [_str_for_input_var(x) for x in self.owner.inputs[3:]]

        return "{} ~ {}({})".format(print_name, dist_name, ", ".join(dist_args))


def _str_for_input_var(var):
    # note we're dispatching both on type(var) and on type(var.owner.op) so cannot
    # use the standard functools.singledispatch
    if isinstance(var, TensorConstant):
        return _str_for_constant(var)
    elif isinstance(var.owner.op, RandomVariable):
        return var.name if var.name is not None else "<unnamed>"
    # elif isinstance(var.owner.op, Elemwise):
    #     return _str_for_elemwise(var)
    elif isinstance(var.owner.op, DimShuffle):
        return _str_for_input_var(var.owner.inputs[0])
    else:
        return _str_for_expression(var)


# _binaryops = {at.neg: "-", at.add: "+", at.mul: "*", at.true_div: "/", at.pow: "^"}
# def _str_for_elemwise(var):
#     precedence = '-+*/^'
#     if len(var.owner.inputs) == 2 and var.owner.op in _binaryops.keys():
#         input_strs = [_str_for_input_var(x) for x in var.owner.inputs]
#         return "(" + " {} ".format(_binaryops[var.owner.op]).join([
#             _str_for_input_var(x) for x in var.owner.inputs]) + ")"
#     elif len(var.owner.inputs) == 1 and var.owner.op is at.neg:
#         return "-({})".format(_str_for_input_var(var.owner.inputs[0]))


def _str_for_constant(var):
    if len(var.data.shape) == 0:
        return f"{var.data:.3g}"
    elif len(var.data.shape) == 1 and var.data.shape[0] == 1:
        return "{:.3g}".format(var.data[0])
    else:
        return "<constant>"


def _str_for_expression(var):
    def _expand(x):
        if x.owner and (not isinstance(x, PrettyPrintingTensorVariable)):
            return reversed(x.owner.inputs)

    parents = [
        x
        for x in walk(nodes=var.owner.inputs, expand=_expand)
        if isinstance(x, PrettyPrintingTensorVariable)
    ]
    names = [x.name for x in parents]
    return "f(" + ", ".join(names) + ")"
