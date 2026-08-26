#   Copyright 2024 - present The PyMC Developers
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


import re
import sys

from collections.abc import Iterable
from functools import partial

import numpy as np
import pytensor.tensor as pt

from pytensor.compile import SharedVariable
from pytensor.compile.ops import ViewOp
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import walk
from pytensor.graph.type import HasShape
from pytensor.printing import (
    FunctionPrinter,
    OperatorPrinter,
    PatternPrinter,
    PPrinter,
    Printer,
    set_precedence,
)
from pytensor.printing import pprint as _pytensor_pprint
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot, Sum
from pytensor.tensor.random.op import RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable
from rich.box import SIMPLE_HEAD
from rich.table import Table

from pymc.logprob.abstract import MeasurableOp
from pymc.model import Model
from pymc.pytensorf import _cheap_eval_mode

__all__ = [
    "model_table",
    "str_for_data_var",
    "str_for_dist",
    "str_for_model",
    "str_for_potential_or_deterministic",
]


def str_for_dist(
    dist: Variable,
    formatting: str = "plain",
    include_params: bool = True,
    named_vars: set[Variable] | None = None,
) -> str:
    """Make a human-readable string representation of a Distribution in a model.

    This can be either LaTeX or plain, optionally with distribution parameter
    values included.
    """
    if named_vars is None:
        named_vars = set()

    dist_op = dist.owner.op

    if include_params:
        try:
            dist_args = dist.owner.op.dist_params(dist.owner)
        except Exception:
            # Can happen with SymbolicRandomVariable without extended_signature
            dist_args = [
                x for x in dist.owner.inputs if not isinstance(x.type, RandomType | NoneTypeT)
            ]

        dist_args_str = [
            _str_for_input_var(a, formatting=formatting, named_vars=named_vars) for a in dist_args
        ]

    if (print_name := getattr(dist_op, "_print_name", None)) is not None:
        dist_name = print_name[formatting == "latex"]
    else:
        dist_name = "Unknown"

    print_name = dist.name

    if "latex" in formatting:
        if print_name is not None:
            print_name = r"\text{" + _latex_escape(print_name.strip("$")) + "}"
            print_name = _format_underscore(print_name)

        if include_params:
            params = ",~".join([d.strip("$") for d in dist_args_str])
            if print_name:
                return rf"${print_name} \sim {dist_name}({params})$"
            else:
                return rf"${dist_name}({params})$"

        else:
            if print_name:
                return rf"${print_name} \sim {dist_name}$"
            else:
                return rf"${dist_name}$"

    else:  # plain
        if include_params:
            params = ", ".join(dist_args_str)
            if print_name:
                return rf"{print_name} ~ {dist_name}({params})"
            else:
                return rf"{dist_name}({params})"
        else:
            if print_name:
                return rf"{print_name} ~ {dist_name}"
            else:
                return dist_name


def str_for_data_var(
    var: Constant | SharedVariable, formatting: str = "plain", include_params: bool = True
) -> str:
    """Make a human-readable string representation of a Data variable in a model."""
    print_name = var.name if var.name is not None else "<unnamed>"

    if include_params:
        value_str = _str_for_constant(var, formatting)
    else:
        value_str = None

    if "latex" in formatting:
        latex_name = r"\text{" + _latex_escape(print_name.strip("$")) + "}"
        latex_name = _format_underscore(latex_name)
        if value_str is not None:
            return rf"${latex_name} = \operatorname{{Data}}({value_str.strip('$')})$"
        else:
            return rf"${latex_name} = \operatorname{{Data}}$"
    else:
        if value_str is not None:
            return rf"{print_name} = Data({value_str})"
        else:
            return rf"{print_name} = Data"


def str_for_model(
    model: Model,
    formatting: str = "plain",
    include_params: bool = True,
    deterministic_exprs: bool = False,
) -> str:
    """Make a human-readable string representation of Model.

    This lists all random variables and their distributions, optionally
    including parameter values.

    Parameters
    ----------
    model
        The model to represent.
    formatting
        Either "plain" or "latex".
    include_params
        Whether to include parameter values.
    deterministic_exprs
        If True, Deterministics and Potentials render their full symbolic
        expression instead of an opaque ``f(inputs)`` placeholder. Traversal
        stops at named model variables, which are rendered by name. Purely a
        graph traversal: no rewrites, compilation, or evaluation happen.
    """
    named_vars: set[Variable] = set()
    named_vars.update(model.data_vars)
    named_vars.update(model.free_RVs)
    named_vars.update(model.observed_RVs)
    named_vars.update(model.deterministics)
    named_vars.update(model.potentials)

    # Wrap functions to avoid confusing typecheckers
    sfd = partial(
        str_for_dist, formatting=formatting, include_params=include_params, named_vars=named_vars
    )
    sfp = partial(
        str_for_potential_or_deterministic,
        formatting=formatting,
        include_params=include_params,
        named_vars=named_vars,
        deterministic_exprs=deterministic_exprs,
    )
    sfdv = partial(str_for_data_var, formatting=formatting, include_params=include_params)

    data_reprs = [sfdv(dv) for dv in model.data_vars]
    free_rv_reprs = [sfd(dist) for dist in model.free_RVs]
    observed_rv_reprs = [sfd(rv) for rv in model.observed_RVs]
    det_reprs = [sfp(dist, dist_name="Deterministic") for dist in model.deterministics]
    potential_reprs = [sfp(pot, dist_name="Potential") for pot in model.potentials]

    var_reprs = data_reprs + free_rv_reprs + det_reprs + observed_rv_reprs + potential_reprs

    if not var_reprs:
        return ""
    if "latex" in formatting:
        var_reprs = [_format_underscore(x) for x in var_reprs]
        formatted = []
        for var_repr in var_reprs:
            if var_repr is None:
                continue
            s = var_repr.strip("$")
            if r"\sim" in s:
                s = s.replace(r"\sim", r"&\sim &", 1)
            else:
                s = s.replace(" = ", " &= &", 1)
            formatted.append(s)
        return r"""$$
            \begin{{array}}{{rcl}}
            {}
            \end{{array}}
            $$""".format("\\\\".join(formatted))
    else:
        sep_pattern = re.compile(r" ([~=]) ")
        names = []
        seps = []
        distrs = []
        for s in var_reprs:
            m = sep_pattern.search(s)
            assert m is not None
            names.append(s[: m.start()])
            seps.append(m.group(1))
            distrs.append(s[m.end() :])
        maxlen = max(len(n) for n in names)
        var_reprs = [f"{n:>{maxlen}} {sep} {d}" for n, sep, d in zip(names, seps, distrs)]
        return "\n".join(var_reprs)


def str_for_potential_or_deterministic(
    var: Variable,
    formatting: str = "plain",
    include_params: bool = True,
    dist_name: str = "Deterministic",
    named_vars: set[Variable] | None = None,
    deterministic_exprs: bool = False,
) -> str:
    """Make a human-readable string representation of a Deterministic or Potential in a model.

    This can be either LaTeX or plain, optionally with distribution parameter
    values included.
    """
    if named_vars is None:
        named_vars = set()

    print_name = var.name if var.name is not None else "<unnamed>"
    sep_plain = "~" if dist_name == "Potential" else "="
    sep_latex = r"\sim" if dist_name == "Potential" else "="
    if deterministic_exprs and include_params:
        expr = _str_for_expression_body(var, formatting=formatting, named_vars=named_vars)
        if "latex" in formatting:
            latex_name = r"\text{" + _latex_escape(print_name.strip("$")) + "}"
            return rf"${latex_name} {sep_latex} {expr}$"
        return rf"{print_name} {sep_plain} {expr}"
    if "latex" in formatting:
        print_name = r"\text{" + _latex_escape(print_name.strip("$")) + "}"
        if include_params:
            return rf"${print_name} {sep_latex} \operatorname{{{dist_name}}}({_str_for_expression(var, formatting=formatting, named_vars=named_vars)})$"
        else:
            return rf"${print_name} {sep_latex} \operatorname{{{dist_name}}}$"
    else:  # plain
        if include_params:
            return rf"{print_name} {sep_plain} {dist_name}({_str_for_expression(var, formatting=formatting, named_vars=named_vars)})"
        else:
            return rf"{print_name} {sep_plain} {dist_name}"


def _str_for_input_var(var: Variable, formatting: str, named_vars: set[Variable]) -> str:
    if isinstance(var, Constant | SharedVariable):
        return _str_for_constant(var, formatting)
    elif var in named_vars or isinstance(var.owner.op, MeasurableOp):
        return _str_for_input_rv(var, formatting)
    elif isinstance(var.owner.op, DimShuffle):
        return _str_for_input_var(var.owner.inputs[0], formatting, named_vars)
    else:
        return _str_for_expression(var, formatting, named_vars)


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


def _str_for_constant(var: Constant | SharedVariable, formatting: str) -> str:
    if isinstance(var, Constant):
        var_data = var.data
        var_type = "constant"
    else:
        var_data = var.get_value()
        var_type = "shared"

    return _str_for_constant_value(var_data, formatting, var_type=var_type)


def _str_for_constant_value(
    var_data: np.ndarray, formatting: str, var_type: str = "constant"
) -> str:
    if len(var_data.shape) == 0:
        return f"{var_data:.3g}"
    elif len(var_data.shape) == 1 and var_data.shape[0] == 1:
        return f"{var_data[0]:.3g}"
    elif "latex" in formatting:
        return rf"\text{{<{var_type}>}}"
    else:
        return rf"<{var_type}>"


def _str_for_expression(var: Variable, formatting: str, named_vars: set[Variable]) -> str:
    def _expand(x):
        if x in named_vars:
            return None
        if x.owner and not isinstance(x.owner.op, MeasurableOp):
            return reversed(x.owner.inputs)

    parents = []
    names = []
    for x in walk(nodes=var.owner.inputs, expand=_expand):
        assert isinstance(x, Variable)
        if x in named_vars:
            if x.name:
                parents.append(x)
                names.append(x.name)
        elif x.owner and isinstance(x.owner.op, MeasurableOp):
            parents.append(x)
            xname = x.name
            if xname is None:
                if (opname := getattr(x.owner.op, "name", None)) is not None:
                    xname = rf"<{opname}>"
            assert xname is not None
            names.append(xname)

    if not names:
        if "latex" in formatting:
            return r"\text{<constant>}"
        else:
            return "<constant>"

    if "latex" in formatting:
        return (
            r"f("
            + ",~".join([_latex_text_format(_latex_escape(n.strip("$"))) for n in names])
            + ")"
        )
    else:
        return r"f(" + ", ".join([n.strip("$") for n in names]) + ")"


class _TransparentFirstInputPrinter(Printer):
    """Render a node as its first input.

    Used for ``ViewOp`` (identity wrappers around deterministics) and
    ``DimShuffle`` (axis manipulation that would add clutter without
    changing the expression a reader cares about).
    """

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        r = pstate.pprinter.process(output.owner.inputs[0], pstate)
        pstate.memo[output] = r
        return r


class _BodyLeafPrinter(Printer):
    """Render named model variables by name, without expanding their graphs."""

    def __init__(self, formatting: str):
        self.formatting = formatting

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        # The leaf condition may match an unnamed ViewOp wrapper whose
        # unwrapped target is the named variable being rendered.
        name = _unwrap_viewops(output).name.strip("$")
        if "latex" in self.formatting:
            r = rf"\text{{{_latex_escape(name)}}}"
        else:
            r = name
        pstate.memo[output] = r
        return r


class _BodyDistPrinter(Printer):
    """Render anonymous distributions inside bodies as distribution calls.

    Delegates to ``str_for_dist`` so an inline prior such as
    ``pm.Normal.dist(0, 1)`` looks like its named counterpart on RV lines,
    instead of leaking graph internals (including a nondeterministic RNG
    memory address) into the repr.
    """

    def __init__(self, formatting: str, named_vars: set[Variable]):
        self.formatting = formatting
        self.named_vars = named_vars

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        r = str_for_dist(output, formatting=self.formatting, named_vars=self.named_vars)
        # str_for_dist wraps latex in $...$ for standalone use; strip for inline bodies
        r = r.strip("$")
        pstate.memo[output] = r
        return r


class _LatexFunctionPrinter(Printer):
    r"""Fallback LaTeX rendering: \operatorname{name}(args)."""

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        op = output.owner.op
        if isinstance(op, Elemwise):
            # Elemwise ops are anonymous wrappers around a scalar op
            name = type(op.scalar_op).__name__.lower()
        else:
            name = getattr(op, "name", None) or type(op).__name__
        name = re.sub(r"\W+", "", str(name)) or "op"
        with set_precedence(pstate):
            args = r",\ ".join(pstate.pprinter.process(i, pstate) for i in output.owner.inputs)
        r = rf"\operatorname{{{name}}}\left({args}\right)"
        pstate.memo[output] = r
        return r


class _BodyConstantPrinter(Printer):
    """Render constants like the rest of the model repr does."""

    def __init__(self, formatting: str):
        self.formatting = formatting

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        data = output.data
        if isinstance(data, np.ndarray):
            r = _str_for_constant_value(data, self.formatting)
        else:
            # e.g. NoneConst placeholders inside RV signatures
            r = str(data)
        pstate.memo[output] = r
        return r


class _OwnerlessLeafPrinter(Printer):
    r"""Render ownerless leaves (e.g. unnamed shared variables) gracefully.

    Mirrors the plain-text rendering (the variable's type), since these
    nodes carry no name or mathematical content of their own.
    """

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        r = rf"\text{{{_latex_escape(str(output.type))}}}"
        pstate.memo[output] = r
        return r


def _is_unary_viewop(r) -> bool:
    return r.owner is not None and isinstance(r.owner.op, ViewOp) and len(r.owner.inputs) == 1


def _is_random_op(r) -> bool:
    # Covers both pytensor RandomVariables and PyMC SymbolicRandomVariables,
    # which also derive from RNGConsumerOp.
    return r.owner is not None and isinstance(r.owner.op, RNGConsumerOp)


def _unwrap_viewops(r):
    """Descend through identity wrappers, but never past a named variable.

    Named Deterministics are themselves unary ``ViewOp`` outputs
    (``view_op(var, name=...)``), so unwrapping must treat any named node as
    a hard boundary.
    """
    while getattr(r, "name", None) is None and _is_unary_viewop(r):
        r = r.owner.inputs[0]
    return r


def _make_plain_body_printer(named_leaf_condition, named_vars: set[Variable]) -> PPrinter:
    """Clone the global pytensor printer with model-aware overrides.

    Inheriting the global printer's registrations gives plain-text coverage of
    many ops for free.

    Anonymous distributions are rendered via ``str_for_dist`` (like named RV
    lines) rather than the inherited raw-RNG rendering, which leaks a
    nondeterministic memory address.

    ``ViewOp`` must be handled condition-based (below the named-leaf rule),
    because named Deterministics are themselves unary ``ViewOp`` outputs that
    a dict-keyed entry would unwrap before the leaf check. ``DimShuffle`` has
    a dict entry in the global printer, so it is overridden in kind; its
    outputs are never named, so it cannot preempt the leaf rule.
    """
    printer = _pytensor_pprint.clone_assign(
        DimShuffle,
        _TransparentFirstInputPrinter(),
    )
    printer = printer.clone_assign(
        lambda pstate, r: _is_random_op(r),
        _BodyDistPrinter("plain", named_vars),
    )
    printer = printer.clone_assign(
        lambda pstate, r: _is_unary_viewop(r),
        _TransparentFirstInputPrinter(),
    )
    printer = printer.clone_assign(
        lambda pstate, r: isinstance(r, Constant),
        _BodyConstantPrinter("plain"),
    )
    return printer.clone_assign(named_leaf_condition, _BodyLeafPrinter("plain"))


def _make_latex_body_printer(named_leaf_condition, named_vars: set[Variable]) -> PPrinter:
    r"""A fresh printer rendering expression bodies as LaTeX.

    Priority is the reverse of assignment order (``assign`` inserts at head).
    Ops without a dedicated registration degrade gracefully to
    ``\\operatorname{name}(args)``.
    """
    printer = PPrinter()
    printer.assign(lambda pstate, r: True, _LatexFunctionPrinter())  # lowest priority
    printer.assign(lambda pstate, r: r.owner.op is pt.exp, FunctionPrinter([r"\exp"]))
    printer.assign(lambda pstate, r: r.owner.op is pt.log, FunctionPrinter([r"\log"]))
    printer.assign(
        lambda pstate, r: r.owner.op is pt.sqrt,
        PatternPrinter((r"\sqrt{%(0)s}",)),
    )
    printer.assign(lambda pstate, r: r.owner.op is pt.sin, FunctionPrinter([r"\sin"]))
    printer.assign(lambda pstate, r: r.owner.op is pt.cos, FunctionPrinter([r"\cos"]))
    printer.assign(lambda pstate, r: r.owner.op is pt.tanh, FunctionPrinter([r"\tanh"]))
    printer.assign(
        lambda pstate, r: isinstance(r.owner.op, Sum),
        PatternPrinter((r"\sum\left(%(0)s\right)",)),
    )
    printer.assign(
        lambda pstate, r: isinstance(r.owner.op, Dot),
        PatternPrinter((r"%(0)s \cdot %(1)s",)),
    )
    printer.assign(
        lambda pstate, r: r.owner.op is pt.true_div,
        PatternPrinter((r"\frac{%(0)s}{%(1)s}",)),
    )
    printer.assign(
        lambda pstate, r: r.owner.op is pt.pow,
        PatternPrinter((r"{%(0)s}^{%(1)s}",)),
    )
    printer.assign(lambda pstate, r: r.owner.op is pt.neg, OperatorPrinter("-", 0, "either"))
    printer.assign(lambda pstate, r: r.owner.op is pt.sub, OperatorPrinter("-", -2, "left"))
    printer.assign(
        lambda pstate, r: r.owner.op is pt.add,
        OperatorPrinter("+", -2, "either"),
    )
    printer.assign(
        lambda pstate, r: r.owner.op is pt.mul,
        OperatorPrinter(r"\cdot", -1, "either"),
    )
    # Anonymous distributions render as distribution calls; without this they
    # would fall through to conditions that assume r.owner is not None and
    # crash on their ownerless rng/size inputs.
    # Ownerless leaves (e.g. unnamed shared variables referenced directly)
    # must terminate here: the operator conditions below assume r.owner.
    printer.assign(lambda pstate, r: r.owner is None, _OwnerlessLeafPrinter())
    printer.assign(
        lambda pstate, r: _is_random_op(r),
        _BodyDistPrinter("latex", named_vars),
    )
    # Condition-based ViewOp handling: must stay below the named-leaf rule,
    # which resolves ViewOp wrappers itself. DimShuffle is dict-keyed because
    # pytensor's own dict entry would otherwise take precedence; DimShuffle
    # outputs are never named, so it cannot preempt the leaf rule.
    printer.assign(lambda pstate, r: _is_unary_viewop(r), _TransparentFirstInputPrinter())
    printer.assign(DimShuffle, _TransparentFirstInputPrinter())
    printer.assign(
        lambda pstate, r: isinstance(r, Constant),
        _BodyConstantPrinter("latex"),
    )
    printer.assign(named_leaf_condition, _BodyLeafPrinter("latex"))  # highest priority
    return printer


def _strip_outer_parens(s: str) -> str:
    while s.startswith("(") and s.endswith(")") and _parens_balanced(s[1:-1]):
        s = s[1:-1]
    return s


def _parens_balanced(s: str) -> bool:
    depth = 0
    for char in s:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _str_for_expression_body(var: Variable, formatting: str, named_vars: set[Variable]) -> str:
    """Render the full symbolic body of an expression graph as text or LaTeX.

    Traversal stops at any variable whose name belongs to a known model
    variable; those render by name since their definitions appear elsewhere
    in the model representation.
    """
    body = var
    while (
        body.owner is not None and isinstance(body.owner.op, ViewOp) and len(body.owner.inputs) == 1
    ):
        body = body.owner.inputs[0]

    named_names = {v.name.strip("$") for v in named_vars if v.name is not None}
    if var.name is not None:
        named_names.discard(var.name)

    def _named_leaf_condition(pstate, r) -> bool:
        r = _unwrap_viewops(r)
        name = getattr(r, "name", None)
        return name is not None and name.strip("$") in named_names

    if "latex" in formatting:
        s = _make_latex_body_printer(_named_leaf_condition, named_vars).process(body)
    else:
        s = _make_plain_body_printer(_named_leaf_condition, named_vars).process(body)
    return _strip_outer_parens(s)


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


def _default_repr_pretty(obj: Variable | Model, p, cycle):
    """Handy plug-in method to instruct IPython-like REPLs to use our str_repr above."""
    # we know that our str_repr does not recurse, so we can ignore cycle
    try:
        if not hasattr(obj, "str_repr"):
            raise AttributeError
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
        import IPython.lib.pretty

        IPython.lib.pretty._repr_pprint(obj, p, cycle)


def _register_ipython_pretty_printers():
    """Register our pretty printer with IPython if it is already loaded.

    Doing this only when IPython is in ``sys.modules`` avoids importing it
    eagerly via pymc — IPython is heavy and irrelevant outside REPLs, and any
    REPL that cares will have imported IPython before pymc.
    """
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return
    try:
        ipython.lib.pretty.for_type(TensorVariable, _default_repr_pretty)
        ipython.lib.pretty.for_type(Model, _default_repr_pretty)
    except AttributeError:
        pass


_register_ipython_pretty_printers()


def _format_underscore(variable: str) -> str:
    """Escapes all unescaped underscores in the variable name for LaTeX representation."""
    return re.sub(r"(?<!\\)_", r"\\_", variable)


def _variable_expression(
    model: Model,
    var: Variable,
    truncate_deterministic: int | None,
    named_vars: set[Variable],
) -> str:
    """Get the expression of a variable in a human-readable format."""
    if var in model.data_vars:
        var_expr = "Data"
    elif var in model.deterministics:
        str_repr = str_for_potential_or_deterministic(var, dist_name="", named_vars=named_vars)
        _, var_expr = str_repr.split(" = ")
        var_expr = var_expr[1:-1]
        if truncate_deterministic is not None and len(var_expr) > truncate_deterministic:
            contents = var_expr[2:-1].split(", ")
            str_len = 0
            for show_n, content in enumerate(contents):
                str_len += len(content) + 2
                if str_len > truncate_deterministic:
                    break
            var_expr = f"f({', '.join(contents[:show_n])}, ...)"
    elif var in model.potentials:
        var_expr = str_for_potential_or_deterministic(
            var, dist_name="Potential", named_vars=named_vars
        ).split(" ~ ")[1]
    else:
        var_expr = str_for_dist(var, named_vars=named_vars).split(" ~ ")[1]
    return var_expr


def _dims_expression(model: Model, var: Variable) -> str:
    """Get the dimensions of a variable in a human-readable format."""

    def _extract_dim_value(var: Variable) -> np.ndarray:
        if isinstance(var, SharedVariable):
            return var.get_value(borrow=True)
        if isinstance(var, Constant):
            return var.data
        return var.eval(mode=_cheap_eval_mode)

    if (dims := model.named_vars_to_dims.get(var.name)) is not None:
        dim_sizes = {dim: _extract_dim_value(model.dim_lengths[dim]) for dim in dims}
        return " × ".join(f"{dim}[{dim_size}]" for dim, dim_size in dim_sizes.items())
    if not isinstance(var.type, HasShape):
        return ""
    shape_values = list(pt.as_tensor(var.shape).eval(mode=_cheap_eval_mode))  # type: ignore[attr-defined]
    return f"[{', '.join(map(str, shape_values))}]" if shape_values else ""


def _model_parameter_count(model: Model) -> int:
    """Count the number of parameters in the model."""
    rv_shapes = model.eval_rv_shapes()  # Includes transformed variables
    return sum(int(np.prod(rv_shapes[free_rv.name])) for free_rv in model.free_RVs)


def model_table(
    model: Model,
    *,
    split_groups: bool = True,
    truncate_deterministic: int | None = None,
    parameter_count: bool = True,
) -> Table:
    """Create a rich table with a summary of the model's variables and their expressions.

    Parameters
    ----------
    model : Model
        The PyMC model to summarize.
    split_groups : bool
        If True, each group of variables (data, free_RVs, deterministics, potentials, observed_RVs)
        will be separated by a section.
    truncate_deterministic : int | None
        If not None, truncate the expression of deterministic variables that go beyond this length.
    parameter_count : bool
        If True, add a row with the total number of parameters in the model.

    Returns
    -------
    Table
        A rich table with the model's variables, their expressions and dims.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm

        from pymc import model_table

        coords = {"subject": range(20), "param": ["a", "b"]}
        with pm.Model(coords=coords) as m:
            x = pm.Data("x", np.random.normal(size=(20, 2)), dims=("subject", "param"))
            y = pm.Data("y", np.random.normal(size=(20,)), dims="subject")

            beta = pm.Normal("beta", mu=0, sigma=1, dims="param")
            mu = pm.Deterministic("mu", pm.math.dot(x, beta), dims="subject")
            sigma = pm.HalfNormal("sigma", sigma=1)

            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="subject")

        table = model_table(m)
        table  # Displays the following table in an interactive environment
        '''
         Variable  Expression         Dimensions
        ─────────────────────────────────────────────────────
              x =  Data               subject[20] × param[2]
              y =  Data               subject[20]

           beta ~  Normal(0, 1)       param[2]
          sigma ~  HalfNormal(0, 1)
                                      Parameter count = 3

             mu =  f(beta)            subject[20]

          y_obs ~  Normal(mu, sigma)  subject[20]
        '''

    Output can be explicitly rendered in a rich console or exported to text, html or svg.

    .. code-block:: python

        from rich.console import Console

        console = Console(record=True)
        console.print(table)
        text_export = console.export_text()
        html_export = console.export_html()
        svg_export = console.export_svg()

    """
    table = Table(
        show_header=True,
        show_edge=False,
        box=SIMPLE_HEAD,
        highlight=False,
        collapse_padding=True,
    )
    table.add_column("Variable", justify="right")
    table.add_column("Expression", justify="left")
    table.add_column("Dimensions")

    groups: tuple[Iterable[Variable], ...]
    if split_groups:
        groups = (
            model.data_vars,
            model.free_RVs,
            model.deterministics,
            model.potentials,
            model.observed_RVs,
        )
    else:
        # Show variables in the order they were defined
        groups = (model.named_vars.values(),)

    named_vars: set[Variable] = set()
    named_vars.update(model.data_vars)
    named_vars.update(model.free_RVs)
    named_vars.update(model.observed_RVs)
    named_vars.update(model.deterministics)
    named_vars.update(model.potentials)

    for group in groups:
        if not group:
            continue

        for var in group:
            var_name = var.name
            sep = f"[b]{' ~' if (var in model.basic_RVs) else ' ='}[/b]"
            var_expr = _variable_expression(model, var, truncate_deterministic, named_vars)
            dims_expr = _dims_expression(model, var)
            table.add_row(var_name + sep, var_expr, dims_expr)

        if parameter_count and (not split_groups or group == model.free_RVs):
            n_parameters = _model_parameter_count(model)
            table.add_row("", "", f"[i]Parameter count = {n_parameters}[/i]")

        table.add_section()

    return table
