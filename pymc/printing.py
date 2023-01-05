#   Copyright 2022- The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import itertools
import textwrap

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from copy import copy
from typing import Optional, Union

import pytensor.tensor as pt

from pytensor.compile.function.types import Function
from pytensor.graph.fg import FunctionGraph
from pytensor.printing import (
    IgnorePrinter,
    OperatorPrinter,
    PatternPrinter,
    PPrinter,
    PrinterState,
)
from pytensor.printing import pprint as pt_pprint
from pytensor.raise_op import Assert
from pytensor.scalar.basic import Add, Cast, Mul
from pytensor.tensor.basic import Join, MakeVector, TensorVariable, Variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import _dot
from pytensor.tensor.random.basic import RandomVariable
from pytensor.tensor.random.var import RandomStateSharedVariable
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor

from pymc.distributions import SymbolicRandomVariable
from pymc.distributions.censored import CensoredRV
from pymc.distributions.mixture import MarginalMixtureRV
from pymc.distributions.timeseries import RandomWalkRV
from pymc.model import Model

PrinterStateType = Union[MutableMapping, PrinterState]


def get_op_name(node, output_latex):
    op_name = getattr(node.op, "_print_name", None) or getattr(node.op, "name", None)

    if isinstance(op_name, (tuple, list)):
        op_name = op_name[int(output_latex)]
    elif output_latex:
        op_name = "\\operatorname{%s}" % op_name

    return op_name


class PreamblePPrinter(PPrinter):
    r"""Pretty printer that displays a preamble.
    Preambles are put into an `OrderedDict` of categories (determined by
    printers that use the preamble).  The order can be set by preempting the
    category names within an `OrderedDict` passed to the constructor via
    the `preamble_dict` keyword.

    The lines accumulated in each category are comma-separated up to a fixed
    length given by `PreamblePPrinter.max_preamble_width`, after which a
    newline is appended and process repeats.

    Example
    -------
    >>> import aesara.tensor as at
    >>> from aeppl.printing import pprint
    >>> srng = at.random.RandomStream()
    >>> X_rv = srng.normal(at.scalar('\\mu'), at.scalar('\\sigma'), name='X')
    >>> print(pprint(X_rv))
    \\mu in R
    \\sigma in R
    X ~ N(\\mu, \\sigma**2),  X in R
    X
    XXX: Not thread-safe!
    """

    max_preamble_width = 40

    def __init__(
        self,
        *args,
        pstate_defaults: Optional[PrinterStateType] = None,
        preamble_dict: Optional[Mapping] = None,
        **kwargs,
    ):
        """Create a `PreamblePPrinter`.

        Parameters
        ----------
        pstate_defaults: dict (optional)
            Default printer state parameters.
        preamble_dict: OrderedDict (optional)
            Default preamble dictionary.  Use this to pre-set the print-out
            ordering of preamble categories/keys.
        """
        super().__init__(*args, **kwargs)
        self.pstate_defaults: PrinterStateType = pstate_defaults or {}
        self.pstate_defaults.setdefault(
            "preamble_dict", OrderedDict() if preamble_dict is None else preamble_dict
        )
        self.printers_dict = dict(pt_pprint.printers_dict)
        self.printers = copy(pt_pprint.printers)
        self._pstate = None

    def create_state(self, pstate: Optional[PrinterStateType]):
        if pstate is None:
            pstate = PrinterState(
                pprinter=self, **{k: copy(v) for k, v in self.pstate_defaults.items()}
            )
        elif isinstance(pstate, Mapping):
            pstate.update({k: copy(v) for k, v in self.pstate_defaults.items()})
            pstate = PrinterState(pprinter=self, **pstate)

        # FIXME: Good old fashioned circular references...
        # We're doing this so that `self.process` will be called correctly
        # accross all code.  (I'm lookin' about you, `DimShufflePrinter`; get
        # your act together.)
        pstate.pprinter._pstate = pstate

        return pstate

    def process(self, r: Variable, pstate: Optional[PrinterStateType] = None):
        pstate = self._pstate
        assert pstate
        return super().process(r, pstate)

    def process_graph(self, inputs, outputs, updates=None, display_inputs=False):
        raise NotImplementedError()  # pragma: no cover

    def __call__(self, *args, latex_env="rcl", latex_label: str = None):
        in_vars = args[0]

        pstate = next(iter(args[1:]), None)
        if isinstance(pstate, (MutableMapping, PrinterState)):
            pstate = self.create_state(args[1])
        elif pstate is None:
            pstate = self.create_state(None)

        if isinstance(in_vars, Function):
            in_vars = in_vars.maker.fgraph

        # This pretty printer needs more information about shapes and inputs,
        # which it gets from a `FunctionGraph`.
        fgraph = None
        out_vars = None
        if isinstance(in_vars, FunctionGraph):
            # We were given a `FunctionGraph` to start with; let's make sure
            # it has the shape information we need.
            fgraph = in_vars
            if not hasattr(fgraph, "shape_feature"):
                shape_feature = ShapeFeature()
                fgraph.attach_feature(shape_feature)
            in_vars = fgraph.inputs
            out_vars = fgraph.outputs
        elif not isinstance(in_vars, (tuple, list)):
            in_vars = [in_vars]

        if fgraph is None:
            memo = {}
            fgraph = FunctionGraph(
                outputs=in_vars,
                features=[ShapeFeature()],
                clone=True,
                memo=memo,
            )
            in_vars = [memo[i] for i in in_vars]
            out_vars = fgraph.outputs

        pstate.fgraph = fgraph

        # TODO: How should this be formatted to better designate
        # the output numbers (in LaTeX, as well)?
        body_strs = []
        for v in out_vars:
            # input variables processed
            body_strs += [super().__call__(v, pstate)]

        latex_out = getattr(pstate, "latex", False)

        comma_str = ", \\," if latex_out else ", "
        newline_str = "\n\\\\\n" if latex_out else "\n"
        indent_str = "  "

        # Let's join all the preamble categories, but split within
        # categories when the joined line is too long.
        preamble_lines = []
        for v in pstate.preamble_dict.values():

            if isinstance(v, Mapping):
                v = list(v.values())

            assert isinstance(v, list)

            if not v:
                continue

            v_new = []
            c_len = l_idx = 0
            for l in v:
                if len(v_new) <= l_idx:
                    c_len = self.max_preamble_width * l_idx
                    v_new.append([l])
                else:
                    v_new[l_idx].append(l)
                c_len += len(l)
                l_idx += int(c_len // self.max_preamble_width > l_idx)

            preamble_lines.append(newline_str.join(comma_str.join(z) for z in v_new))

        if preamble_lines and latex_out:
            preamble_body = newline_str.join(preamble_lines + body_strs)
            preamble_str = (
                f"\\begin{{gathered}}\n{textwrap.indent(preamble_body, indent_str)}"
                f"\n\\end{{gathered}}"
            )
            res = newline_str.join([preamble_str])
        else:
            res = newline_str.join(preamble_lines + body_strs)

        if latex_out and latex_env is None:
            return res

        if latex_out and latex_env:
            label_out = f"\\label{{{latex_label}}}\n" if latex_label else ""
            res = textwrap.indent(res, indent_str)
            res = f"\\begin{{{latex_env}}}\n" f"{res}\n" f"{label_out}" f"\\end{{{latex_env}}}"

        return res


class GenericSubtensorPrinter:
    def process(self, r: Variable, pstate: Optional[PrinterStateType]):
        if getattr(r, "owner", None) is None:  # pragma: no cover
            raise TypeError("Can only print `*Subtensor*`s.")

        output_latex = getattr(pstate, "latex", False)

        inputs = list(r.owner.inputs)
        obj = inputs.pop(0)
        idxs = getattr(r.owner.op, "idx_list", inputs)
        sidxs = []
        old_precedence = getattr(pstate, "precedence", None)
        try:
            pstate.precedence = -1000

            for entry in idxs:
                if isinstance(entry, slice):
                    s_parts = [""] * 2
                    if entry.start is not None:
                        s_parts[0] = pstate.pprinter.process(inputs.pop())

                    if entry.stop is not None:
                        s_parts[1] = pstate.pprinter.process(inputs.pop())

                    if entry.step is not None:
                        s_parts.append(pstate.pprinter.process(inputs.pop()))

                    sidxs.append(":".join(s_parts))
                else:
                    sidxs.append(pstate.pprinter.process(inputs.pop()))

            if output_latex:
                idx_str = ", \\,".join(sidxs)
            else:
                idx_str = ", ".join(sidxs)
        finally:
            pstate.precedence = old_precedence

        try:
            pstate.precedence = 1000
            sub = pstate.pprinter.process(obj, pstate)
        finally:
            pstate.precedence = old_precedence

        if output_latex:
            return f"{sub}\\left[{idx_str}\\right]"
        else:
            return f"{sub}[{idx_str}]"


class PyMCVariablePrinter:
    r"""Pretty print PyMC variables.
    `Op`\s are able to specify their ascii and LaTeX formats via a "print_name"
    property.  `Op.print_name` should be a tuple or list that specifies the
    plain text/ascii and LaTeX name, respectively.

    Note that this class was originally taken from aeppl/printing.py and bears the
    name RandomVariablePrinter there.
    """

    def __init__(self, input_idx_start: int = 3):
        """Create a `PyMCVariablePrinter`.
        Parameters
        ----------
        name: str (optional)
            A fixed name to use for the random variables printed by this
            printer.  If not specified, use `RandomVariable.name`.
        """
        self.input_idx_start = input_idx_start

    def split_tilde(self, p_str_repr):
        if " ~ " in p_str_repr:
            return p_str_repr.split(" ~ ")[1]
        if r" \sim " in p_str_repr:
            return p_str_repr.split(r" \sim ")[1]
        return p_str_repr

    def handle_input_params(self, p, pprinter, pstate):
        """
        Equivalent of previous printing `_str_for_input_var`.
        """
        if getattr(p, "name", None):
            return "\\text{" + p.name + "}"

        f_param = pprinter.process(p, pstate)

        try:
            f_param = f"{float(f_param):2g}".strip()
        except ValueError:
            pass

        return self.split_tilde(f_param)

    def process(self, output, pstate: Optional[PrinterStateType]):
        if hasattr(pstate, "memo") and output in pstate.memo:
            return pstate.memo[output]

        pprinter = pstate.pprinter
        node = getattr(output, "owner", None)

        output_latex = getattr(pstate, "latex", False)
        op_name = get_op_name(node, output_latex)

        if op_name is None:  # pragma: no cover
            raise ValueError(f"Could not find a name for {node.op}")

        preamble_dict = getattr(pstate, "preamble_dict", {})
        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, "precedence", None)
            pstate.precedence = new_precedence

            # Get the symbol name string from another pprinter.
            # We create a dummy variable with no `owner`, so that
            # the pprinter will format it like a standard variable.
            dummy_out = output.clone()
            dummy_out.owner = None
            # Use this to get shape information down the line.
            dummy_out.orig_var = output

            var_name = pprinter.process(dummy_out, pstate)
            if output_latex:
                dist_format = "\\text{%s} \\sim %s\\left(%s\\right)"
            else:
                dist_format = "%s ~ %s(%s)"

            # Get the shape info for our dummy symbol, if available,
            # and append it to the distribution definition.
            # TODO: Propagate this change upstream in Aesara's pretty printer.
            if "shape_strings" in preamble_dict:
                shape_info_str = preamble_dict["shape_strings"].pop(dummy_out)
                shape_info_str = shape_info_str.lstrip(var_name)
                if output_latex:
                    dist_format += f"\\, {shape_info_str}"
                else:
                    dist_format += shape_info_str

            dist_params = node.inputs[self.input_idx_start :]

            formatted_params = []
            for p in dist_params:

                """
                Hackish solution: if we can retrieve the parameter name, we
                use that. Otherwise, we revert to pprinter.process.
                """
                f_param = self.handle_input_params(p, pprinter, pstate)

                if f_param is not None:
                    # e.g. GaussianRandomWalk has init_dist=None by default
                    formatted_params.append(f_param)

            dist_def_str = dist_format % (
                var_name,
                op_name,
                ", ".join(formatted_params),
            )

        finally:
            pstate.precedence = old_precedence

        # All subsequent calls will use the variable name and
        # not the distribution definition.
        pstate.memo[output] = var_name

        if preamble_dict:
            rv_strings = preamble_dict.setdefault("rv_strings", [])
            rv_strings.append(dist_def_str)
            return var_name
        else:
            return dist_def_str


class SymbolicVariablePrinter(PyMCVariablePrinter):
    """
    Generic printer for symbolic distributions. Custom printers should
    be defined due to 1) symbolic random graphs being very different
    from one distribution to another and 2) what parameters are to be
    printed can vary as well.
    """

    def __init__(self, input_idx_start: int = 2):
        super().__init__(input_idx_start)

    def handle_input_params(self, p, pprinter, pstate):
        return self.split_tilde(pprinter.process(p, pstate))


class RandomWalkPrinter(SymbolicVariablePrinter):
    def __init__(self, input_idx_start: int = 0):
        super().__init__(input_idx_start)

    def handle_input_params(self, p, pprinter, pstate):
        if (
            getattr(p, "owner", None)
            and len(p.owner.inputs) == 1
            and isinstance(p.owner.op, Elemwise)
            and isinstance(p.owner.op.scalar_op, Cast)
        ):
            p = p.owner.inputs[0]

            # steps parameter
            while getattr(p, "owner", None) and isinstance(
                p.owner.op, (Join, MakeVector, Subtensor, AdvancedSubtensor, AdvancedSubtensor1)
            ):
                # steps parameter being broadcasted many times
                p = p.owner.inputs[0]

        return super().handle_input_params(p, pprinter, pstate)


class MarginalMixturePrinter:
    def __init__(self, input_idx_start: int = 1):
        super().__init__(input_idx_start)


def str_for_model(model, formatting="latex"):
    all_vars = itertools.chain(model.unobserved_RVs, model.observed_RVs, model.potentials)
    if formatting == "latex":
        rv_reprs = [latex_pprint(var, latex_env=None) for var in all_vars]
        rv_reprs = [
            rv_repr.replace(r"\sim", r"&\sim&").replace(", ", ",~")
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

    elif formatting == "plain":
        rv_reprs = [pprint(var) for var in all_vars]
        names = [s[: s.index("~") - 1] for s in rv_reprs]
        distrs = [s[s.index("~") + 2 :] for s in rv_reprs]
        maxlen = str(max(len(x) for x in names))
        rv_reprs = [
            ("{name:>" + maxlen + "} ~ {distr}").format(name=n, distr=d)
            for n, d in zip(names, distrs)
        ]
        return "\n".join(rv_reprs)


pprint = PreamblePPrinter()

# Handles printing of any `RandomVariable``
pprint.assign(
    lambda pstate, r: getattr(r, "owner", None) and isinstance(r.owner.op, RandomVariable),
    PyMCVariablePrinter(),
)

"""
Here, printers for `SymbolicRandomVariable`s are assigned.
First, a generic printer for `SymbolicRandomVariable`s is determined
such that, when a new symbolic distribution is created, its pretty
print does not crash nor yield the default pointer of the instance.

Specific printers tailored to the symbolic distribution are then assigned.
"""

pprint.assign(
    lambda pstate, r: getattr(r, "owner", None) and isinstance(r.owner.op, SymbolicRandomVariable),
    SymbolicVariablePrinter(),
)

pprint.assign(CensoredRV, SymbolicVariablePrinter(input_idx_start=0))
pprint.assign(MarginalMixtureRV, SymbolicVariablePrinter(input_idx_start=1))
pprint.assign(RandomWalkRV, RandomWalkPrinter())


# This handles the in-place versions of `Add` and `Mul` produced by
# rewrites
pprint.assign(
    lambda pstate, r: getattr(r, "owner", None)
    and isinstance(r.owner.op, Elemwise)
    and isinstance(r.owner.op.scalar_op, Add),
    OperatorPrinter("+", -1, "left"),
)
pprint.assign(
    lambda pstate, r: getattr(r, "owner", None)
    and isinstance(r.owner.op, Elemwise)
    and isinstance(r.owner.op.scalar_op, Mul),
    OperatorPrinter("*", -1, "left"),
)

pprint.assign(_dot, OperatorPrinter("@", -1, "left"))
pprint.assign(pt.and_, OperatorPrinter("and", -1, "left"))
pprint.assign(pt.or_, OperatorPrinter("or", -1, "left"))
pprint.assign(Assert, IgnorePrinter())
pprint.assign(RandomStateSharedVariable, IgnorePrinter())

subtensor_printer = GenericSubtensorPrinter()
pprint.assign(Subtensor, subtensor_printer)
pprint.assign(AdvancedSubtensor, subtensor_printer)
pprint.assign(AdvancedSubtensor1, subtensor_printer)

pprint.assign(pt.ge, PatternPrinter(("%(0)s >= %(1)s", -1000)))
pprint.assign(pt.gt, PatternPrinter(("%(0)s > %(1)s", -1000)))
pprint.assign(pt.le, PatternPrinter(("%(0)s <= %(1)s", -1000)))
pprint.assign(pt.lt, PatternPrinter(("%(0)s < %(1)s", -1000)))
pprint.assign(pt.eq, PatternPrinter(("%(0)s == %(1)s", -1000)))

latex_pprint = PreamblePPrinter(pstate_defaults={"latex": True})
latex_pprint.assign(Assert, IgnorePrinter())
latex_pprint.assign(RandomStateSharedVariable, IgnorePrinter())
latex_pprint.printers = copy(pprint.printers)
latex_pprint.printers_dict = dict(pprint.printers_dict)

latex_pprint.assign(pt.ge, PatternPrinter(("%(0)s \\ge %(1)s", -1000)))
latex_pprint.assign(pt.gt, PatternPrinter(("%(0)s \\gt %(1)s", -1000)))
latex_pprint.assign(pt.le, PatternPrinter(("%(0)s \\le %(1)s", -1000)))
latex_pprint.assign(pt.lt, PatternPrinter(("%(0)s \\lt %(1)s", -1000)))
latex_pprint.assign(pt.eq, PatternPrinter(("%(0)s = %(1)s", -1000)))

latex_pprint.assign(pt.and_, OperatorPrinter("\\land", -1, "left"))
latex_pprint.assign(pt.or_, OperatorPrinter("\\lor", -1, "left"))
latex_pprint.assign(pt.invert, PatternPrinter(("\\lnot %(0)s", -1000)))

latex_pprint.assign(_dot, OperatorPrinter("\\;", -1, "left"))
latex_pprint.assign(pt.mul, OperatorPrinter("\\odot", -1, "either"))
latex_pprint.assign(pt.true_div, PatternPrinter(("\\frac{%(0)s}{%(1)s}", -1000)))
latex_pprint.assign(pt.sqrt, PatternPrinter(("\\sqrt{%(0)s}", -1000)))
latex_pprint.assign(pt.pow, PatternPrinter(("{%(0)s}^{%(1)s}", -1000)))


def _default_repr_pretty(obj: Union[TensorVariable, Model], p, cycle):
    """Handy plug-in method to instruct IPython-like REPLs to use our str_repr above."""
    # we know that our str_repr does not recurse, so we can ignore cycle
    try:
        output = pprint(obj)
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
