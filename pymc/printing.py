#   Copyright 2021 The PyMC Developers
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

from aesara.graph.basic import walk
from aesara.tensor.basic import TensorVariable, Variable
from aesara.tensor.elemwise import DimShuffle
from aesara.tensor.random.basic import RandomVariable
from aesara.tensor.var import TensorConstant

from pymc.model import Model

__all__ = [
    "str_for_dist",
    "str_for_model",
    "str_for_potential_or_deterministic",
]


def str_for_dist(rv: TensorVariable, formatting: str = "plain", include_params: bool = True) -> str:
    """Make a human-readable string representation of a RandomVariable in a model, either
    LaTeX or plain, optionally with distribution parameter values included."""

    if include_params:
        # first 3 args are always (rng, size, dtype), rest is relevant for distribution
        dist_args = [_str_for_input_var(x, formatting=formatting) for x in rv.owner.inputs[3:]]

    print_name = rv.name if rv.name is not None else "<unnamed>"
    if "latex" in formatting:
        print_name = r"\text{" + _latex_escape(print_name) + "}"
        dist_name = rv.owner.op._print_name[1]
        if include_params:
            return r"${} \sim {}({})$".format(print_name, dist_name, ",~".join(dist_args))
        else:
            return rf"${print_name} \sim {dist_name}$"
    else:  # plain
        dist_name = rv.owner.op._print_name[0]
        if include_params:
            return r"{} ~ {}({})".format(print_name, dist_name, ", ".join(dist_args))
        else:
            return rf"{print_name} ~ {dist_name}"


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
        print_name = r"\text{" + _latex_escape(print_name) + "}"
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
    def _is_potential_or_determinstic(var: Variable) -> bool:
        try:
            return var.str_repr.__func__.func is str_for_potential_or_deterministic
        except AttributeError:
            # in case other code overrides str_repr, fallback
            return False

    if isinstance(var, TensorConstant):
        return _str_for_constant(var, formatting)
    elif isinstance(var.owner.op, RandomVariable) or _is_potential_or_determinstic(var):
        # show the names for RandomVariables, Deterministics, and Potentials, rather
        # than the full expression
        return _str_for_input_rv(var, formatting)
    elif isinstance(var.owner.op, DimShuffle):
        return _str_for_input_var(var.owner.inputs[0], formatting)
    else:
        return _str_for_expression(var, formatting)


def _str_for_input_rv(var: Variable, formatting: str) -> str:
    _str = var.name if var.name is not None else "<unnamed>"
    if "latex" in formatting:
        return r"\text{" + _latex_escape(_str) + "}"
    else:
        return _str


def _str_for_constant(var: TensorConstant, formatting: str) -> str:
    if len(var.data.shape) == 0:
        return f"{var.data:.3g}"
    elif len(var.data.shape) == 1 and var.data.shape[0] == 1:
        return f"{var.data[0]:.3g}"
    elif "latex" in formatting:
        return r"\text{<constant>}"
    else:
        return r"<constant>"


def _str_for_expression(var: Variable, formatting: str) -> str:
    # construct a string like f(a1, ..., aN) listing all random variables a as arguments
    def _expand(x):
        if x.owner and (not isinstance(x.owner.op, RandomVariable)):
            return reversed(x.owner.inputs)

    parents = [
        x
        for x in walk(nodes=var.owner.inputs, expand=_expand)
        if x.owner and isinstance(x.owner.op, RandomVariable)
    ]
    names = [x.name for x in parents]

    if "latex" in formatting:
        return r"f(" + ",~".join([r"\text{" + _latex_escape(n) + "}" for n in names]) + ")"
    else:
        return r"f(" + ", ".join(names) + ")"


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
