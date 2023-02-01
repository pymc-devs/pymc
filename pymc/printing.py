#   Copyright 2023 The PyMC Developers
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

from typing import Union

from pytensor.compile import SharedVariable
from pytensor.graph.basic import Constant, walk
from pytensor.tensor.basic import TensorVariable, Variable
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.basic import RandomVariable
from pytensor.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)

from pymc.model import Model

__all__ = [
    "str_for_dist",
    "str_for_model",
    "str_for_potential_or_deterministic",
]


def str_for_dist(
    dist: TensorVariable, formatting: str = "plain", include_params: bool = True
) -> str:
    """Make a human-readable string representation of a Distribution in a model, either
    LaTeX or plain, optionally with distribution parameter values included."""

    if include_params:
        # first 3 args are always (rng, size, dtype), rest is relevant for distribution
        if isinstance(dist.owner.op, RandomVariable):
            dist_args = [
                _str_for_input_var(x, formatting=formatting) for x in dist.owner.inputs[3:]
            ]
        else:
            dist_args = [
                _str_for_input_var(x, formatting=formatting)
                for x in dist.owner.inputs
                if not isinstance(x, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
            ]

    print_name = dist.name

    if "latex" in formatting:
        if print_name is not None:
            print_name = r"\text{" + _latex_escape(dist.name.strip("$")) + "}"

        op_name = (
            dist.owner.op._print_name[1]
            if hasattr(dist.owner.op, "_print_name")
            else r"\\operatorname{Unknown}"
        )
        if include_params:
            if print_name:
                return r"${} \sim {}({})$".format(
                    print_name, op_name, ",~".join([d.strip("$") for d in dist_args])
                )
            else:
                return r"${}({})$".format(op_name, ",~".join([d.strip("$") for d in dist_args]))

        else:
            if print_name:
                return rf"${print_name} \sim {op_name}$"
            else:
                return rf"${op_name}$"

    else:  # plain
        dist_name = (
            dist.owner.op._print_name[0] if hasattr(dist.owner.op, "_print_name") else "Unknown"
        )
        if include_params:
            if print_name:
                return r"{} ~ {}({})".format(print_name, dist_name, ", ".join(dist_args))
            else:
                return r"{}({})".format(dist_name, ", ".join(dist_args))
        else:
            if print_name:
                return rf"{print_name} ~ {dist_name}"
            else:
                return dist_name


def str_for_model(model: Model, formatting: str = "plain", include_params: bool = True) -> str:
    """Make a human-readable string representation of Model, listing all random variables
    and their distributions, optionally including parameter values."""
    all_rv = itertools.chain(model.unobserved_RVs, model.observed_RVs, model.potentials)

    rv_reprs = [rv.str_repr(formatting=formatting, include_params=include_params) for rv in all_rv]
    rv_reprs = [rv_repr for rv_repr in rv_reprs if "TransformedDistribution()" not in rv_repr]

    if not rv_reprs:
        return ""
    if "latex" in formatting:
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
        # align vars on their ~
        names = [s[: s.index("~") - 1] for s in rv_reprs]
        distrs = [s[s.index("~") + 2 :] for s in rv_reprs]
        maxlen = str(max(len(x) for x in names))
        rv_reprs = [
            ("{name:>" + maxlen + "} ~ {distr}").format(name=n, distr=d)
            for n, d in zip(names, distrs)
        ]
        return "\n".join(rv_reprs)


def str_for_potential_or_deterministic(
    var: TensorVariable,
    formatting: str = "plain",
    include_params: bool = True,
    dist_name: str = "Deterministic",
) -> str:
    """Make a human-readable string representation of a Deterministic or Potential in a model, either
    LaTeX or plain, optionally with distribution parameter values included."""
    print_name = var.name if var.name is not None else "<unnamed>"
    if "latex" in formatting:
        print_name = r"\text{" + _latex_escape(print_name.strip("$")) + "}"
        if include_params:
            return rf"${print_name} \sim \operatorname{{{dist_name}}}({_str_for_expression(var, formatting=formatting)})$"
        else:
            return rf"${print_name} \sim \operatorname{{{dist_name}}}$"
    else:  # plain
        if include_params:
            return rf"{print_name} ~ {dist_name}({_str_for_expression(var, formatting=formatting)})"
        else:
            return rf"{print_name} ~ {dist_name}"


def _str_for_input_var(var: Variable, formatting: str) -> str:
    # Avoid circular import
    from pymc.distributions.distribution import SymbolicRandomVariable

    def _is_potential_or_deterministic(var: Variable) -> bool:
        try:
            return var.str_repr.__func__.func is str_for_potential_or_deterministic
        except AttributeError:
            # in case other code overrides str_repr, fallback
            return False

    if isinstance(var, (Constant, SharedVariable)):
        return _str_for_constant(var, formatting)
    elif isinstance(
        var.owner.op, (RandomVariable, SymbolicRandomVariable)
    ) or _is_potential_or_deterministic(var):
        # show the names for RandomVariables, Deterministics, and Potentials, rather
        # than the full expression
        return _str_for_input_rv(var, formatting)
    elif isinstance(var.owner.op, DimShuffle):
        return _str_for_input_var(var.owner.inputs[0], formatting)
    else:
        return _str_for_expression(var, formatting)


def _str_for_input_rv(var: Variable, formatting: str) -> str:
    _str = (
        var.name
        if var.name is not None
        else str_for_dist(var, formatting=formatting, include_params=True)
    )
    if "latex" in formatting:
        return _latex_text_format(_latex_escape(_str.strip("$")))
    else:
        return _str


def _str_for_constant(var: Union[Constant, SharedVariable], formatting: str) -> str:
    if isinstance(var, Constant):
        var_data = var.data
        var_type = "constant"
    else:
        var_data = var.get_value()
        var_type = "shared"

    if len(var_data.shape) == 0:
        return f"{var_data:.3g}"
    elif len(var_data.shape) == 1 and var_data.shape[0] == 1:
        return f"{var_data[0]:.3g}"
    elif "latex" in formatting:
        return rf"\text{{<{var_type}>}}"
    else:
        return rf"<{var_type}>"


def _str_for_expression(var: Variable, formatting: str) -> str:
    # Avoid circular import
    from pymc.distributions.distribution import SymbolicRandomVariable

    # construct a string like f(a1, ..., aN) listing all random variables a as arguments
    def _expand(x):
        if x.owner and (not isinstance(x.owner.op, (RandomVariable, SymbolicRandomVariable))):
            return reversed(x.owner.inputs)

    parents = [
        x
        for x in walk(nodes=var.owner.inputs, expand=_expand)
        if x.owner and isinstance(x.owner.op, (RandomVariable, SymbolicRandomVariable))
    ]
    names = [x.name for x in parents]

    if "latex" in formatting:
        return (
            r"f("
            + ",~".join([_latex_text_format(_latex_escape(n.strip("$"))) for n in names])
            + ")"
        )
    else:
        return r"f(" + ", ".join([n.strip("$") for n in names]) + ")"


def _latex_text_format(text: str) -> str:
    if r"\operatorname{" in text:
        return text
    else:
        return r"\text{" + text + "}"


def _latex_escape(text: str) -> str:
    # Note that this is *NOT* a proper LaTeX escaper, on purpose. _repr_latex_ is
    # primarily used in the context of Jupyter notebooks, which render using MathJax.
    # MathJax is a subset of LaTeX proper, which expects only $ to be escaped. If we were
    # to also escape e.g. _ (replace with \_), then "\_" will show up in the output, etc.
    return text.replace("$", r"\$")


def _default_repr_pretty(obj: Union[TensorVariable, Model], p, cycle):
    """Handy plug-in method to instruct IPython-like REPLs to use our str_repr above."""
    # we know that our str_repr does not recurse, so we can ignore cycle
    try:
        output = obj.str_repr()
        # Find newlines and replace them with p.break_()
        # (see IPython.lib.pretty._repr_pprint)
        lines = output.splitlines()
        with p.group():
            for idx, output_line in enumerate(lines):
                if idx:
                    p.break_()
                p.text(output_line)
    except AttributeError:
        # the default fallback option (no str_repr method)
        IPython.lib.pretty._repr_pprint(obj, p, cycle)


try:
    # register our custom pretty printer in ipython shells
    import IPython

    IPython.lib.pretty.for_type(TensorVariable, _default_repr_pretty)
    IPython.lib.pretty.for_type(Model, _default_repr_pretty)
except (ModuleNotFoundError, AttributeError):
    # no ipython shell
    pass
