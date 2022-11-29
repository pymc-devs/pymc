import string
import textwrap
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from copy import copy
from typing import Optional, Union

import aesara
import aesara.tensor as at
from aesara.compile.function.types import Function
from aesara.graph.basic import Constant, Variable
from aesara.graph.fg import FunctionGraph
from aesara.printing import (
    IgnorePrinter,
    OperatorPrinter,
    PatternPrinter,
    PPrinter,
    PrinterState,
)
from aesara.printing import pprint as at_pprint
from aesara.raise_op import Assert
from aesara.scalar.basic import Add, Mul
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import _dot
from aesara.tensor.random.basic import NormalRV
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.var import RandomStateSharedVariable
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.subtensor import AdvancedSubtensor, AdvancedSubtensor1, Subtensor
from aesara.tensor.type import float_dtypes, int_dtypes, uint_dtypes
from aesara.tensor.var import TensorConstant, TensorVariable

try:
    from sympy import Array as SympyArray
    from sympy.printing import latex as sympy_latex

    def latex_print_array(data):  # pragma: no cover
        return sympy_latex(SympyArray(data))

except ImportError:  # pragma: no cover

    def latex_print_array(data):
        return data


PrinterStateType = Union[MutableMapping, PrinterState]


class RandomVariablePrinter:
    r"""Pretty print random variables.

    `Op`\s are able to specify their ascii and LaTeX formats via a "print_name"
    property.  `Op.print_name` should be a tuple or list that specifies the
    plain text/ascii and LaTeX name, respectively.

    Also, distribution parameters can be formatted distinctly by overriding
    the `RandomVariablePrinter.process_param` method.

    """

    def __init__(self, name: Optional[str] = None):
        """Create a `RandomVariablePrinter`.

        Parameters
        ----------
        name: str (optional)
            A fixed name to use for the random variables printed by this
            printer.  If not specified, use `RandomVariable.name`.

        """
        self.name = name

    def process_param(self, idx: int, sform: str, pstate: Optional[PrinterStateType]):
        """Perform special per-parameter post-formatting.

        This can be used, for instance, to change a std. dev. into a variance.

        Parameters
        ----------
        idx: int
            The index value of the parameter.
        sform: str
            The pre-formatted string form of the parameter.
        pstate: object
            The printer state.

        """
        return sform  # pragma: no cover

    def process(self, output, pstate: Optional[PrinterStateType]):
        if hasattr(pstate, "memo") and output in pstate.memo:
            return pstate.memo[output]

        pprinter = pstate.pprinter
        node = getattr(output, "owner", None)

        if node is None or not isinstance(node.op, RandomVariable):  # pragma: no cover
            raise TypeError(
                "Function %s cannot represent a variable that is "
                "not the result of a RandomVariable operation" % self.name
            )

        op_name = self.name or getattr(node.op, "_print_name", None)
        op_name = op_name or getattr(node.op, "name", None)

        if op_name is None:  # pragma: no cover
            raise ValueError(f"Could not find a name for {node.op}")

        # Allow `Op`s to specify their ascii and LaTeX formats (in a tuple/list
        # with that order).
        output_latex = getattr(pstate, "latex", False)
        if isinstance(op_name, (tuple, list)):
            op_name = op_name[int(output_latex)]
        elif output_latex:
            op_name = "\\operatorname{%s}" % op_name

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
                dist_format = "%s \\sim %s\\left(%s\\right)"
            else:
                dist_format = "%s ~ %s(%s)"

            # Get the shape info for our dummy symbol, if available,
            # and append it to the distribution definition.
            # TODO: Propagate this change upstream in Aesara's pretty printer.
            if "shape_strings" in preamble_dict:
                shape_info_str = preamble_dict["shape_strings"].pop(dummy_out)
                shape_info_str = shape_info_str.lstrip(var_name)
                if output_latex:
                    dist_format += "\\, {}".format(shape_info_str)
                else:
                    dist_format += shape_info_str

            dist_params = node.inputs[3:]
            formatted_params = [
                self.process_param(i, pprinter.process(p, pstate), pstate)
                for i, p in enumerate(dist_params)
            ]

            # We remove trailing zeros and limit the number of decimals
            # on floats
            formatted_params = []
            for i, p in enumerate(dist_params):
                f_param = self.process_param(i, pprinter.process(p, pstate), pstate)
                try:
                    f_param = f"{float(f_param):2g}".strip()
                except ValueError:
                    pass
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
            return "%s\\left[%s\\right]" % (sub, idx_str)
        else:
            return "%s[%s]" % (sub, idx_str)


class VariableWithShapePrinter:
    """Print variable shape info in the preamble.

    Also uses readable character names for un-named variables.

    Constant arrays are only printed when their size is below a threshold
    set by ``max_line_width * max_line_height``

    """

    available_names = OrderedDict.fromkeys(string.ascii_letters)
    default_printer = aesara.printing.default_printer
    max_line_width = 40
    max_line_height = 20

    @classmethod
    def process(cls, output: Variable, pstate: Optional[PrinterStateType]):
        if output in pstate.memo:
            return pstate.memo[output]

        using_latex = getattr(pstate, "latex", False)
        # Crude--but effective--means of stopping print-outs for large
        # arrays.
        constant = isinstance(
            output, (TensorConstant, aesara.scalar.basic.ScalarConstant)
        )
        too_large = constant and (
            output.data.size > cls.max_line_width * cls.max_line_height
        )

        if constant and not too_large:
            # Print constants that aren't too large
            if using_latex and output.ndim > 0:
                out_name = latex_print_array(output.data)
            else:
                out_name = str(output.data)
        elif (
            isinstance(
                output,
                (
                    TensorVariable,
                    aesara.scalar.basic.ScalarType,
                    aesara.scalar.basic.ScalarVariable,
                ),
            )
            or constant
        ):
            # Process name and shape

            # Attempt to get the original variable, in case this is a cloned
            # `RandomVariable` output; otherwise, we won't get any shape
            # information from the `FunctionGraph`.
            var = getattr(output, "orig_var", output)

            out_name = cls.process_variable_name(var, pstate)

            shape_info = cls.process_shape_info(var, pstate)

            shape_strings = pstate.preamble_dict.setdefault(
                "shape_strings", OrderedDict()
            )
            shape_strings[output] = shape_info
        else:  # pragma: no cover
            raise TypeError(f"Type {type(output)} not handled by variable printer")

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_variable_name(
        cls, output: Variable, pstate: Optional[PrinterStateType]
    ):
        """Take a variable name from the available ones.

        This function also initializes the available names by removing
        all the manually specified names within the `FunctionGraph`
        being printed (if available). Doing so removes the potential for
        name collisions.

        """
        if output in pstate.memo:
            return pstate.memo[output]

        available_names = getattr(pstate, "available_names", None)
        if available_names is None:
            # Initialize this state's available names
            available_names = copy(cls.available_names)
            # Remove known names in the graph.
            _ = [available_names.pop(v.name, None) for v in pstate.fgraph.variables]
            setattr(pstate, "available_names", available_names)

        if getattr(output, "name", None):
            # Observed an existing name; remove it.
            out_name = output.name
            available_names.pop(out_name, None)
        else:
            # Take an unused name.
            out_name, _ = available_names.popitem(last=False)

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_shape_info(cls, output: Variable, pstate: Optional[PrinterStateType]):
        using_latex = getattr(pstate, "latex", False)

        if output.dtype in int_dtypes:
            sspace_char = "Z"
        elif output.dtype in uint_dtypes:
            sspace_char = "N"
        elif output.dtype in float_dtypes:
            sspace_char = "R"
        else:
            sspace_char = "?"

        shape_feature = None
        if not hasattr(pstate.fgraph, "shape_feature"):
            pstate.fgraph.attach_feature(ShapeFeature())
        shape_feature = pstate.fgraph.shape_feature

        shape_dims = []
        for i in range(output.ndim):
            s_i_out = None
            if using_latex:
                s_i_pat = "N^{%s}" + ("_{%s}" % i)
            else:
                s_i_pat = "N^%s" + ("_%s" % i)
            if shape_feature:
                new_precedence = -1000
                try:
                    old_precedence = getattr(pstate, "precedence", None)
                    pstate.precedence = new_precedence
                    _s_i_out = shape_feature.get_shape(output, i)

                    if not isinstance(_s_i_out, (Constant, TensorVariable)):
                        s_i_out = pstate.pprinter.process(_s_i_out, pstate)
                    else:
                        s_i_out = str(at.get_scalar_constant_value(_s_i_out))

                except (KeyError, IndexError, ValueError, NotScalarConstantError):
                    # Ugh, most of these exception types are just for
                    # `get_scalar_constant_value`!
                    # TODO: The design of that function contract could use some
                    # serious reconsideration.
                    pass
                finally:
                    pstate.precedence = old_precedence

            if not s_i_out:
                s_i_out = cls.process_variable_name(output, pstate)
                s_i_out = s_i_pat % s_i_out

            shape_dims += [s_i_out]

        shape_info = cls.process_variable_name(output, pstate)
        if using_latex:
            shape_info += " \\in \\mathbb{%s}" % sspace_char
            shape_dims_str = " \\times ".join(shape_dims)
            if shape_dims_str:
                shape_info += "^{%s}" % shape_dims_str
        else:
            shape_info += " in %s" % sspace_char
            shape_dims_str = " x ".join(shape_dims)
            if shape_dims:
                shape_info += "**(%s)" % shape_dims_str

        return shape_info


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
    >>> X_rv = at.random.normal(at.scalar('\\mu'), at.scalar('\\sigma'), name='X')
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
        self.printers_dict = dict(at_pprint.printers_dict)
        self.printers = copy(at_pprint.printers)
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

    def __call__(self, *args, latex_env="equation", latex_label: str = None):
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

        if latex_out and latex_env:
            label_out = f"\\label{{{latex_label}}}\n" if latex_label else ""
            res = textwrap.indent(res, indent_str)
            res = (
                f"\\begin{{{latex_env}}}\n"
                f"{res}\n"
                f"{label_out}"
                f"\\end{{{latex_env}}}"
            )

        return res


pprint = PreamblePPrinter()

# The order here is important!
pprint.printers.insert(
    0,
    (
        lambda pstate, r: isinstance(r, (aesara.scalar.basic.ScalarType, Variable)),
        VariableWithShapePrinter,
    ),
)
pprint.printers.insert(
    0,
    (
        lambda pstate, r: getattr(r, "owner", None)
        and isinstance(r.owner.op, RandomVariable),
        RandomVariablePrinter(),
    ),
)

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


class NormalRVPrinter(RandomVariablePrinter):
    def __init__(self):
        super().__init__("N")

    def process_param(self, idx, sform, pstate):
        if idx == 1:
            if getattr(pstate, "latex", False):
                return f"{{{sform}}}^{{2}}"
            else:
                return f"{sform}**2"
        else:
            return sform


pprint.assign(NormalRV, NormalRVPrinter())
pprint.assign(_dot, OperatorPrinter("@", -1, "left"))
pprint.assign(at.and_, OperatorPrinter("and", -1, "left"))
pprint.assign(at.or_, OperatorPrinter("or", -1, "left"))
pprint.assign(Assert, IgnorePrinter())
pprint.assign(RandomStateSharedVariable, IgnorePrinter())
# pprint.assign(random_state_type, IgnorePrinter())

subtensor_printer = GenericSubtensorPrinter()
pprint.assign(Subtensor, subtensor_printer)
pprint.assign(AdvancedSubtensor, subtensor_printer)
pprint.assign(AdvancedSubtensor1, subtensor_printer)

pprint.assign(at.ge, PatternPrinter(("%(0)s >= %(1)s", -1000)))
pprint.assign(at.gt, PatternPrinter(("%(0)s > %(1)s", -1000)))
pprint.assign(at.le, PatternPrinter(("%(0)s <= %(1)s", -1000)))
pprint.assign(at.lt, PatternPrinter(("%(0)s < %(1)s", -1000)))
pprint.assign(at.eq, PatternPrinter(("%(0)s == %(1)s", -1000)))

latex_pprint = PreamblePPrinter(pstate_defaults={"latex": True})
latex_pprint.assign(Assert, IgnorePrinter())
latex_pprint.assign(RandomStateSharedVariable, IgnorePrinter())
latex_pprint.printers = copy(pprint.printers)
latex_pprint.printers_dict = dict(pprint.printers_dict)

latex_pprint.assign(at.ge, PatternPrinter(("%(0)s \\ge %(1)s", -1000)))
latex_pprint.assign(at.gt, PatternPrinter(("%(0)s \\gt %(1)s", -1000)))
latex_pprint.assign(at.le, PatternPrinter(("%(0)s \\le %(1)s", -1000)))
latex_pprint.assign(at.lt, PatternPrinter(("%(0)s \\lt %(1)s", -1000)))
latex_pprint.assign(at.eq, PatternPrinter(("%(0)s = %(1)s", -1000)))

latex_pprint.assign(at.and_, OperatorPrinter("\\land", -1, "left"))
latex_pprint.assign(at.or_, OperatorPrinter("\\lor", -1, "left"))
latex_pprint.assign(at.invert, PatternPrinter(("\\lnot %(0)s", -1000)))

latex_pprint.assign(_dot, OperatorPrinter("\\;", -1, "left"))
latex_pprint.assign(at.mul, OperatorPrinter("\\odot", -1, "either"))
latex_pprint.assign(at.true_div, PatternPrinter(("\\frac{%(0)s}{%(1)s}", -1000)))
latex_pprint.assign(at.sqrt, PatternPrinter(("\\sqrt{%(0)s}", -1000)))
latex_pprint.assign(at.pow, PatternPrinter(("{%(0)s}^{%(1)s}", -1000)))
