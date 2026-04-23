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

from collections.abc import Iterable
from functools import partial

import numpy as np
import pytensor.tensor as pt

from pytensor.compile import SharedVariable
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import walk
from pytensor.graph.type import HasShape
from pytensor.tensor.elemwise import DimShuffle
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


def str_for_model(model: Model, formatting: str = "plain", include_params: bool = True) -> str:
    """Make a human-readable string representation of Model.

    This lists all random variables and their distributions, optionally
    including parameter values.
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
        IPython.lib.pretty._repr_pprint(obj, p, cycle)


try:
    # register our custom pretty printer in ipython shells
    import IPython

    IPython.lib.pretty.for_type(TensorVariable, _default_repr_pretty)
    IPython.lib.pretty.for_type(Model, _default_repr_pretty)
except (ModuleNotFoundError, AttributeError):
    # no ipython shell
    pass


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
