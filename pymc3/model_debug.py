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

import numpy as np
import theano
import theano.tensor as tt
from theano.graph.basic import graph_inputs
from theano.graph.toolbox import is_same_graph

from pymc3.model import modelcontext


def find_explicit_bounds(apply, _bound_nodes=None):
    """
    Recursively trasverse the nodes downstream of an apply, looking for explicit
    bound elemwise nodes (tt.switch) and bound scalar nodes (tt.all)

    Parameters
    ----------
    apply: Theano apply node from which to search recursively
    _bound_nodes: Output list (internal)

    Returns
    -------
    List containing tuples of (bound node first input, bound node type (elemwise or scalar))

    """

    def is_switch(apply_op):
        return hasattr(apply_op, "scalar_op") and isinstance(
            apply_op.scalar_op, theano.scalar.basic.Switch
        )

    def is_all(apply_op):
        return hasattr(apply_op, "scalar_op") and isinstance(
            apply_op.scalar_op, theano.scalar.basic.AND
        )

    # Initialize output list at first level of iteration
    if _bound_nodes is None:
        _bound_nodes = []

    # Check if it is a switch node
    if is_switch(apply.op) and apply.out.name == "elemwise_bound_switch":
        _bound_nodes.append((apply.inputs[0], "elemwise"))

    elif is_all(apply.op) and apply.out.name == 'scalar_bound':
        _bound_nodes.append((apply.inputs[0], "scalar"))

    # Call function recursively on non terminal inputs
    for apply_input in apply.inputs:
        child_apply = apply_input.owner
        if child_apply:
            find_explicit_bounds(child_apply, _bound_nodes)

    return _bound_nodes


def find_nested_logical_conds(apply, _logical_conds=None):
    """
    Find logical comparison operations downstream of apply node

    Parameters
    ----------
    apply: Theano apply node from which to search recursively
    _logical_conds: Output list (internal)

    Returns
    -------
    list of logical comparison apply nodes

    """
    def is_logical_comparison(apply_op):
        return hasattr(apply_op, "scalar_op") and isinstance(
            apply_op.scalar_op, theano.scalar.basic.LogicalComparison
        )

    #     # Initialize output list at first level of iteration
    if _logical_conds is None:
        _logical_conds = []

    # if hasattr(apply.op, "scalar_op") and isinstance(apply.op.scalar_op, theano.scalar.basic.LogicalComparison):
    if is_logical_comparison(apply.op):
        _logical_conds.append(apply)

    # Call recursively on non terminal inputs
    for child_node in apply.inputs:
        child_apply = child_node.owner
        if child_apply:
            find_nested_logical_conds(child_apply, _logical_conds)

    return _logical_conds


def find_logical_cond_input_variables(logical_cond_apply):
    """
    Extract graph_inputs to logical cond apply node.

    A minimal parsing of constant variables that do not affect apply node output is attempted.

    Parameters
    ----------
    logical_cond_apply: Apply node of logical comparison
        Expects an apply with maximum of two input nodes

    Returns
    -------
    List of input variables (minus parsing of some constants)

    """
    # TODO: Better parsing

    inputs = logical_cond_apply.inputs
    if len(inputs) > 2:
        raise ValueError

    input_variables = []
    for expression in inputs:
        if not expression.owner:
            input_variables.append((expression,))
        else:
            expression_inputs = tuple(graph_inputs(expression.owner.inputs))

            # Try to find 1.0 * var expression introduced by the bound_switches
            # in order to remove Constant(1) from output
            found_mul_1 = False
            for potential_var in expression_inputs:
                test_expression = 1.0 * potential_var
                if is_same_graph(expression, test_expression):
                    input_variables.append((potential_var,))
                    found_mul_1 = True
            # Otherwise, include all inputs
            if not found_mul_1:
                input_variables.append(expression_inputs)

    return input_variables


def input_variables_to_string(ivs, mask=None):
    """
    Get string representiation of graph input variable and respective (masked) values

    Single variables are represented as var 1 = values
    Multiple variables are represented as f(var 1 = values, var 2 = values)

    For constants only the string representation of the (masked) values is returned
    If values cannot be extracted, only the string representation of the variable is returned

    Parameters
    ----------
    ivs: list of input variables
    mask: mask to apply to the extracted values of all input variabels
        Defaults to None

    Returns
    -------
    String representation of input variables and (masked) values
    """
    # TODO: Less hackish way to mask values ?

    def apply_mask(vals):
        if mask is None:
            return vals
        return np.squeeze(np.unique(np.resize(vals, mask.shape)[mask]))

    def get_name_value_from_iv(iv):
        # Constant variable
        if hasattr(iv, "value"):
            return apply_mask(iv.value)
        # Theano variable
        if hasattr(iv, "tag"):
            return f"{iv} = {apply_mask(iv.tag.test_value)}"
        # Something else
        return iv

    if len(ivs) == 1:
        return str(get_name_value_from_iv(ivs[0]))
    else:
        return f"f({', '.join(map(str, map(get_name_value_from_iv, ivs)))})"


def check_bounds(model=None, variable=None, mask_good_inputs=True):
    """
    Check whether explicit bound checks in the logp or logcdf methods of Pymc3 distributions
    are being violated, leading to -inf probability / bad energy.

    Parameters
    ----------
    model: Pymc3 model
        model for which to check_bounds. If None, model is inferred from context stack
        or from variable (if possible)
    variable: Pymc3 variable (or string name if model is available)
        variable for which to check_bounds. If None, all basic_RVs in the model which
        generate -inf for model.test_point are checked. Note that only untransformed
        variables are tested. You can set transform=None to make sure bound checking is
        done on untransformed space.
    mask_good_inputs: Bool
        Whether to mask values that are not violating the bound checks

    Returns
    -------
    None
    """
    # TODO: Get parameter / observed names for more useful output (is this possible)?
    # TODO: Add logic for BinaryBitOps for combined logical expressions (e.g., tt.and() and tt.or())?

    theano_logical_comparators_parse_map = {
        theano.scalar.basic.EQ: "==",
        theano.scalar.basic.NEQ: "!=",
        theano.scalar.basic.GT: ">",
        theano.scalar.basic.GE: ">=",
        theano.scalar.basic.LT: "<",
        theano.scalar.basic.LE: "<=",
    }

    # Try to get model from context stack or variable
    if model is None:
        try:
            model = modelcontext(None)
        except TypeError:
            if hasattr(variable, "model"):
                model = modelcontext(variable.model)
            else:
                raise ValueError("Cannot determine model context from inputs")

    # Check if check_bounds is enabled
    if not model.check_bounds:
        raise ValueError(
            "check_bounds was set to False during the initialization of the Model.\n"
            "Rebuild the model, while making sure that check_bounds is set to True\n"
            "in order to make use of check_bounds."
        )

    # If no variable was specified, call function for every variable that has a -inf initial evaluation
    if variable is None:
        for variable in model.basic_RVs:
            if not np.isfinite(variable.logp(model.test_point)):
                check_bounds(model, variable, mask_good_inputs)
        return

    # Check that variable exists in model
    if isinstance(variable, str):
        variable = model.named_vars.get(variable)
    if variable not in model.basic_RVs:
        raise ValueError(
            f"Variable is not present in model context. Model variables include: {model.basic_RVs}\n"
            "You may have tried to check the bounds of a transformed variable. If this is the case\n"
            "you can temporarily set transform=None when specifying the distribution in question"
        )

    triggers_inf = False
    untriggers_inf_op = tt.eq

    # Test bounds for variable
    no_result = True
    for bound_switch_input, bound_switch_type in find_explicit_bounds(variable.logpt.owner):

        # Check that -inf switch is triggered for at least one input (otherwise it cannot be responsible)
        bound_fn = model.fn(bound_switch_input)
        bound_output = bound_fn(model.test_point)
        if not np.any(bound_output == triggers_inf):
            continue

        # If bound is responsible, find nested logical conditions that could be responsible
        if bound_switch_type == 'elemwise':
            bound_logical_conds = find_nested_logical_conds(bound_switch_input.owner)
        else:
            # Skip the first neq operation in scalar bound switches
            bound_logical_conds = find_nested_logical_conds(bound_switch_input.owner.inputs[0].owner)

        # Sanity check that disabling all logical conditions untriggers -inf switch
        # TODO: Remove this?
        no_bound = theano.clone(
            bound_switch_input,
            {
                logical_cond.out: untriggers_inf_op(logical_cond.inputs[0], logical_cond.inputs[0])
                for logical_cond in bound_logical_conds
            },
        )
        no_bound_fn = model.fn(no_bound)
        no_bound_output = no_bound_fn(model.test_point)
        if np.any(no_bound_output == triggers_inf):
            raise RuntimeWarning("Disabling all bounds is not working as expected")

        # Enable one logical condition at a time (assume culprit if switch is still triggered)
        for enabled_logical_cond in bound_logical_conds:

            new_bound = theano.clone(
                bound_switch_input,
                {
                    logical_cond.out: untriggers_inf_op(
                        logical_cond.inputs[0], logical_cond.inputs[0]
                    )
                    for logical_cond in bound_logical_conds
                    if logical_cond != enabled_logical_cond
                },
            )
            new_bound_fn = model.fn(new_bound)
            new_bound_output = new_bound_fn(model.test_point)

            # Print information about triggering condition
            if np.any(new_bound_output == triggers_inf):
                if no_result:
                    no_result = False
                    print(f"The following explicit bound(s) of {variable} were violated:")

                mask = None
                if mask_good_inputs:
                    # We can use the new_bound_fn directly to mask culprits
                    if bound_switch_type == 'elemwise':
                        mask = new_bound_output == triggers_inf
                    # We have to check the output of the enabled condition directly
                    # as the output of new_bound_fn is collapsed via tt.all
                    else:
                        cond_fn = model.fn(enabled_logical_cond.out)
                        cond_fn_output = cond_fn(model.test_point)
                        mask = cond_fn_output == triggers_inf

                ivs = []
                for iv in find_logical_cond_input_variables(enabled_logical_cond):
                    # TODO: Add fallback in case masking fails
                    ivs.append(input_variables_to_string(iv, mask))

                logical_cond_type = type(enabled_logical_cond.op.scalar_op)
                logical_comp = theano_logical_comparators_parse_map.get(logical_cond_type, str(logical_cond_type))

                print(f"{ivs[0]} {logical_comp} {ivs[1]}")

    if no_result:
        print(
            f"No explicit bounds of {variable} were violated for the given inputs,",
            "An infinite logp could have arised from one of the following:",
            "   1. Undefined arithmetic operations (e.g., 1/0)",
            "   2. Numerical precision issues",
            "   3. Implicit bounds in the logp expression",
            sep="\n",
        )
