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

import itertools
import re

from aesara.graph.basic import walk
from aesara.tensor.basic import TensorVariable
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import RandomVariable
from aesara.tensor.var import TensorConstant


class PrettyPrintingTensorVariable(TensorVariable):
    def _str_repr(self, formatting="plain"):
        print_name = self.name if self.name is not None else "<unnamed>"
        dist_name = self.owner.op._print_name[0]

        if "latex" in formatting:
            print_name = r"\text{" + _latex_escape(print_name) + "}"
            dist_name = self.owner.op._print_name[1]

        # first 3 args are always (rng, size, dtype), rest is relevant for distribution
        dist_args = [_str_for_input_var(x, formatting=formatting) for x in self.owner.inputs[3:]]

        if "latex" in formatting:
            return r"${} \sim {}({})$".format(print_name, dist_name, ",~".join(dist_args))
        else:
            return r"{} ~ {}({})".format(print_name, dist_name, ", ".join(dist_args))

    def __latex__(self, formatting="latex", **kwargs):
        return self._str_repr(formatting=formatting)

    def __str__(self, formatting="plain"):
        return self._str_repr(formatting=formatting)

    _repr_latex_ = __latex__


def str_for_model(model, formatting="plain"):
    all_rv = itertools.chain(model.unobserved_RVs, model.observed_RVs)

    if "latex" in formatting:
        rv_reprs = [rv.__latex__(formatting=formatting) for rv in all_rv]
        rv_reprs = [
            rv_repr.replace(r"\sim", r"&\sim &").strip("$")
            for rv_repr in rv_reprs
            if rv_repr is not None
        ]
        return r"""$$
            \begin{{array}}{{rcl}}
            {}
            \end{{array}}
            $$""".format(
            "\\\\".join(rv_reprs)
        )
    else:
        rv_reprs = [rv.__str__() for rv in all_rv]
        rv_reprs = [rv_repr for rv_repr in rv_reprs if "TransformedDistribution()" not in rv_repr]
        # align vars on their ~
        names = [s[: s.index("~") - 1] for s in rv_reprs]
        distrs = [s[s.index("~") + 2 :] for s in rv_reprs]
        maxlen = str(max(len(x) for x in names))
        rv_reprs = [
            ("{name:>" + maxlen + "} ~ {distr}").format(name=n, distr=d)
            for n, d in zip(names, distrs)
        ]
        return "\n".join(rv_reprs)


def _str_for_input_var(var, formatting):
    # note we're dispatching both on type(var) and on type(var.owner.op) so cannot
    # use the standard functools.singledispatch
    if isinstance(var, TensorConstant):
        return _str_for_constant(var, formatting)
    elif isinstance(var.owner.op, RandomVariable):
        return _str_for_randomvariable(var, formatting)
    elif isinstance(var.owner.op, DimShuffle):
        return _str_for_input_var(var.owner.inputs[0], formatting)
    else:
        return _str_for_expression(var, formatting)


def _str_for_randomvariable(var, formatting):
    _str = var.name if var.name is not None else "<unnamed>"
    if "latex" in formatting:
        return r"\text{" + _latex_escape(_str) + "}"
    else:
        return _str


def _str_for_constant(var, formatting):
    if len(var.data.shape) == 0:
        return f"{var.data:.3g}"
    elif len(var.data.shape) == 1 and var.data.shape[0] == 1:
        return "{:.3g}".format(var.data[0])
    elif "latex" in formatting:
        return r"\text{\textless{}constant\textgreater{}}"
    else:
        return r"<constant>"


def _str_for_expression(var, formatting):
    # construct a string like f(a1, ..., aN) listing all random variables a as arguments
    def _expand(x):
        if x.owner and (not isinstance(x, PrettyPrintingTensorVariable)):
            return reversed(x.owner.inputs)

    parents = [
        x
        for x in walk(nodes=var.owner.inputs, expand=_expand)
        if isinstance(x, PrettyPrintingTensorVariable)
    ]
    names = [x.name for x in parents]

    if "latex" in formatting:
        return r"f(" + ",~".join([r"\text{" + _latex_escape(n) + "}" for n in names]) + ")"
    else:
        return r"f(" + ", ".join(names) + ")"


def _latex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
        "\\": r"\textbackslash{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    regex = re.compile(
        "|".join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: -len(item)))
    )
    return regex.sub(lambda match: conv[match.group()], text)
