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

import functools
import itertools
import warnings

from typing import List, Optional, Tuple, Union

from pytensor.compile import SharedVariable
from pytensor.graph.basic import Constant, walk
from pytensor.tensor.basic import TensorVariable, Variable
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.basic import RandomVariable
from pytensor.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)

from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.model import Model

__all__ = ["str_for_model", "str_for_model_var"]


def str_for_model_var(
    var: TensorVariable, formatting: str = "plain", dist_name: Optional[str] = None, **kwargs
) -> str:
    """Make a human-readable string representation of a Model variable.

    Intended for Distribution, Deterministic, and Potential.
    """
    if not (
        _has_owner(var) and isinstance(var.owner.op, (RandomVariable, SymbolicRandomVariable))
    ) and not _is_potential_or_deterministic(var):
        raise ValueError(
            f"Variable for pretty-printing must be a model variable or the output of .dist(). Received unsupported variable {var}"
        )
    var_name, dist_name, args_str = _get_varname_distname_args(
        var, formatting=formatting, dist_name=dist_name
    )
    if "include_params" in kwargs:
        warnings.warn(
            "The `include_params` argument has been deprecated. Future versions will always include parameter values.",
            FutureWarning,
            stacklevel=2,
        )
        # When removing, remove **kwargs from here and str_for_model().
        if not kwargs["include_params"]:
            args_str = ""
    if formatting == "latex":
        out = rf"${var_name} \sim {dist_name}({args_str})$"
    elif formatting == "plain":
        var_name = var_name.replace("~", "-")
        dist_name = dist_name.replace("~", "-")
        args_str = args_str.replace("~", "-")
        out = f"{var_name} ~ {dist_name}({args_str})"
    else:
        raise ValueError(
            f"Formatting method not recognized: {formatting}. Supported formatting: [latex, plain]."
        )
    return out


def str_for_model(model: Model, formatting: str = "plain", **kwargs) -> str:
    """Make a human-readable string representation of Model including all random
    variables and their distributions.
    """
    all_rv = itertools.chain(model.unobserved_RVs, model.observed_RVs, model.potentials)
    rv_reprs = [rv.str_repr(formatting=formatting, **kwargs) for rv in all_rv]
    if not rv_reprs:
        return ""
    if formatting == "latex":
        rv_reprs = [rv_repr.replace(r"\sim", r"&\sim &").strip("$") for rv_repr in rv_reprs]
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


def _get_varname_distname_args(
    var: TensorVariable, formatting: str, dist_name: Optional[str] = None
) -> Tuple[str, str, str]:
    """Generate formatted strings for the name, distribution name, and
    arguments list of a Model variable.

    For Distribution, Potential, Deterministic, or .dist().
    """
    # Name and distribution name
    name = var.name if var.name is not None else "<unnamed>"  # May be missing if from a dist()
    if (
        not dist_name
        and _has_owner(var)
        and hasattr(var.owner.op, "_print_name")
        and var.owner.op._print_name
    ):
        # The _print_name tuple is necessary for maximum prettiness because a few RVs
        # use special formatting (e.g. superscripts) for their latex print name
        dist_name = (
            var.owner.op._print_name[1] if formatting == "latex" else var.owner.op._print_name[0]
        )
    elif not dist_name:
        raise ValueError(
            f"Missing distribution name for model variable: {var}. Provide one via the"
            " _print_name attribute of your RandomVariable."
        )
    if formatting == "latex":
        name = _latex_clean_command(name, command="text")
        dist_name = _latex_clean_command(dist_name, command="operatorname")

    # Arguments passed to the distribution or expression
    if _has_owner(var) and isinstance(var.owner.op, RandomVariable):
        # var is the RV or dist() from a Distribution.
        dist_args = var.owner.inputs[3:]  # First 3 inputs are always rng, size, dtype
    elif _has_owner(var) and isinstance(var.owner.op, SymbolicRandomVariable):
        # var is a symbolic RV from a Distribution.
        dist_args = [
            x
            for x in var.owner.inputs
            if not isinstance(x, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
        ]
    else:
        # Assume that var is a Deterministic or a Potential.
        dist_args = _walk_expression_args(var)
    args_str = _str_for_args_list(dist_args, formatting=formatting)
    if _is_potential_or_deterministic(var):
        args_str = f"f({args_str})"  # TODO do we still want to do this?

    # These three strings are now formatted according to `formatting`. If latex, they
    # just ultimately need to be wrapped in $.
    return name, dist_name, args_str


def _str_for_input_var(var: Variable, formatting: str) -> str:
    """Make a human-readable string representation for a variable that
    serves as input to another variable."""
    # Check for constants first, because they won't have var.owner
    if isinstance(var, (Constant, SharedVariable)):
        # Give the constant value if it's small enough, else basic type info.
        # Previously _str_for_constant()
        if isinstance(var, Constant):
            var_data = var.data
            var_type = "constant"
        else:
            var_data = var.get_value()
            var_type = "shared"
        if var_data.size == 1:
            return f"{var_data.flatten()[0]:.3g}"
        else:
            return f"<{var_type} {var_data.shape}>"
    elif _has_owner(var):
        if isinstance(var.owner.op, DimShuffle):
            # Recurse
            return _str_for_input_var(var.owner.inputs[0], formatting=formatting)
        elif _is_potential_or_deterministic(var) or isinstance(
            var.owner.op, (RandomVariable, SymbolicRandomVariable)
        ):
            # Give the name of the RV/Potential/Deterministic if available
            if var.name:
                return var.name
            # But if rv comes from .dist() we print the distribution with its args
            else:
                _, dist_name, args_str = _get_varname_distname_args(var, formatting=formatting)
                return f"{dist_name}({args_str})"
        else:
            # Return an "expression" i.e. indicate that this variable is a function of other
            # variables. Looks like f(arg1, ..., argN). Previously _str_for_expression()
            args = _walk_expression_args(var)
            args_str = _str_for_args_list(args, formatting=formatting)
            return f"f({args_str})"
    else:
        raise ValueError(
            f"Unidentified variable in dist or expression args: {var}. If you think this is a bug, please create an issue in the project Github."
        )


def _walk_expression_args(var: Variable) -> List[Variable]:
    """Find all arguments of an expression"""
    if not var.owner:
        return []

    def _expand(x):
        if x.owner and (not isinstance(x.owner.op, (RandomVariable, SymbolicRandomVariable))):
            return reversed(x.owner.inputs)

    return [
        x
        for x in walk(nodes=var.owner.inputs, expand=_expand)
        if x.owner and isinstance(x.owner.op, (RandomVariable, SymbolicRandomVariable))
    ]


def _str_for_args_list(args: List[Variable], formatting: str) -> str:
    """Create a human-readable string representation for the list of inputs
    to a distribution or expression."""
    strs = [_str_for_input_var(x, formatting=formatting) for x in args]
    if formatting == "latex":
        # Format the str as \text{} only if it hasn't been formatted yet and it isn't numeric
        strs_formatted = [
            s
            if r"\text" in s or r"\operatorname" in s or s == "f()" or s.replace(".", "").isdigit()
            else _latex_clean_command(s, command="text")
            for s in strs
        ]
        return ",~".join(strs_formatted)
    else:
        return ", ".join(strs)


def _latex_clean_command(text: str, command: str) -> str:
    r"""Prepare text for LaTeX and maybe wrap it in a \command{}."""
    text = text.replace("$", r"\$")
    # str_for_model() uses \sim to format the array, and properly
    # tilde in latex is hard. So we replace for simplicity
    text = text.replace("~", "-")
    if not text.startswith(rf"\{command}"):
        # The printing module is designed such that text never passes through this
        # function more than once. However, in some cases the text may have already
        # been formatted for LaTeX -- such as when accessing a RandomVariable's
        # pre-specified _print_name. In these cases we avoid the double wrap.
        text = rf"\{command}{{{text}}}"
    # This escape is a workaround for pymc#6508. MathJax is the latex engine in Jupyter
    # notebooks, but its behavior in \text commands deviates from canonical LaTeX: stuff
    # in the \text block will be rendered more or less verbatim. Meanwhile, other
    # engines such as KaTeX behave differently and expect certain characters to be
    # escaped to be typeset as desired. We work around this by escaping out of the
    # command itself, writing the character, then continuing on with the same command.
    if command == "text":
        text = text.replace("_", rf"}}\_\{command}{{")
    return text


def _is_potential_or_deterministic(var: Variable) -> bool:
    # This is a bit hacky but seems like the best we got
    if (
        hasattr(var, "str_repr")
        and callable(var.str_repr)
        and isinstance(var.str_repr.__func__, functools.partial)
    ):
        args = [*var.str_repr.__func__.args, *var.str_repr.__func__.keywords.values()]
        return "Deterministic" in args or "Potential" in args
    return False


def _has_owner(var: Variable):
    return hasattr(var, "owner") and var.owner


def _pymc_pprint(obj: Union[TensorVariable, Model], *args, **kwargs):
    """Pretty-print method that instructs IPython to use our `str_repr()`.

    Note that `str_repr()` is assigned in the initialization functions for
    the objects of interest, i.e. Distribution and Model.
    """
    if hasattr(obj, "str_repr") and callable(obj.str_repr):
        s = obj.str_repr()
    else:
        s = repr(obj)
    # Allow IPython to deal with newlines and the actual printing to shell
    IPython.lib.pretty._repr_pprint(s, *args, **kwargs)


try:
    # register our custom pretty printer in ipython shells
    import IPython

    IPython.lib.pretty.for_type(TensorVariable, _pymc_pprint)
    IPython.lib.pretty.for_type(Model, _pymc_pprint)
except (ModuleNotFoundError, AttributeError):
    # no ipython shell
    pass
