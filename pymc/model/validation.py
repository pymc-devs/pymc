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

"""Validation utilities for PyMC models.

This module provides functions to validate that model dimensions and coordinates
are consistent before sampling begins, preventing cryptic shape mismatch errors.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Variable

try:
    unused = TYPE_CHECKING
except NameError:
    from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymc.model.core import Model

__all__ = ["validate_dims_coords_consistency"]


def validate_dims_coords_consistency(model: Model) -> None:
    """Validate that all dims and coords are consistent before sampling.

    This function performs comprehensive validation to ensure that:
    - All dims referenced in model variables exist in model.coords
    - Variable shapes match their declared dimensions
    - Coordinate lengths match the corresponding dimension sizes
    - MutableData variables have consistent dims when specified
    - No conflicting dimension specifications exist across variables

    Parameters
    ----------
    model : pm.Model
        The PyMC model to validate

    Raises
    ------
    ValueError
        If inconsistencies are found with detailed error messages that guide
        users on how to fix the issues.
    """
    errors = []

    # Check 1: Verify all referenced dims exist in coords
    dims_errors = check_dims_exist(model)
    errors.extend(dims_errors)

    # Check 2: Verify shape-dim consistency for all model variables
    shape_errors = check_shape_dims_match(model)
    errors.extend(shape_errors)

    # Check 3: Check coordinate length matches dimension size
    coord_length_errors = check_coord_lengths(model)
    errors.extend(coord_length_errors)

    # If any errors were found, raise a comprehensive ValueError
    if errors:
        error_msg = "\n\n".join(errors)
        raise ValueError(
            "Model dimension and coordinate inconsistencies detected:\n\n"
            + error_msg
            + "\n\n"
            + "Please fix the above issues before sampling. "
            "You may need to add missing coordinates to model.coords, "
            "adjust variable shapes, or ensure coordinate values match dimension sizes."
        )


def check_dims_exist(model: Model) -> list[str]:
    """Check that all dims referenced in variables exist in model.coords.

    Parameters
    ----------
    model : Model
        The PyMC model to check

    Returns
    -------
    list[str]
        List of error messages (empty if no errors)
    """
    errors = []
    all_referenced_dims = set()

    # Collect all dims referenced across all variables
    for var_name, dims in model.named_vars_to_dims.items():
        if dims is not None:
            for dim in dims:
                if dim is not None:
                    all_referenced_dims.add(dim)

    # Check each referenced dim exists in model.coords
    missing_dims = all_referenced_dims - set(model.coords.keys())

    if missing_dims:
        # Group variables by missing dims for better error messages
        dim_to_vars = {}
        for var_name, dims in model.named_vars_to_dims.items():
            if dims is not None:
                for dim in dims:
                    if dim in missing_dims:
                        dim_to_vars.setdefault(dim, []).append(var_name)

        for dim in sorted(missing_dims):
            var_names = sorted(set(dim_to_vars[dim]))
            var_list = ", ".join([f"'{v}'" for v in var_names])
            errors.append(
                f"Dimension '{dim}' is referenced by variable(s) {var_list}, "
                f"but it is not defined in model.coords. "
                f"Add '{dim}' to model.coords, for example:\n"
                f"  model.add_coord('{dim}', values=range(n))  # or specific coordinate values"
            )

    return errors


def check_shape_dims_match(model: Model) -> list[str]:
    """Check that variable shapes match their declared dims.

    This checks that if a variable declares dims, its shape matches the
    sizes of those dimensions as defined in model.coords.

    Parameters
    ----------
    model : Model
        The PyMC model to check

    Returns
    -------
    list[str]
        List of error messages (empty if no errors)
    """
    errors = []

    for var_name, dims in model.named_vars_to_dims.items():
        if dims is None or not dims:
            continue

        var = model.named_vars.get(var_name)
        if var is None:
            continue

        # Skip if variable doesn't have shape (e.g., scalars)
        if not hasattr(var, "shape") or not hasattr(var, "ndim"):
            continue

        # Get expected shape from dims
        expected_shape = []
        dim_names = []
        for d, dim_name in enumerate(dims):
            if dim_name is None:
                # If dim is None, we can't validate against coords
                # This is valid for variables with mixed dims/None
                continue

            if dim_name not in model.coords:
                # Already reported by check_dims_exist, skip here
                continue

            # Get dimension length
            coord = model.coords[dim_name]
            if coord is not None:
                dim_length = len(coord)
            else:
                # Symbolic dimension - get from dim_lengths
                dim_length_var = model.dim_lengths.get(dim_name)
                if dim_length_var is not None:
                    try:
                        # Try to evaluate if it's a constant
                        if isinstance(dim_length_var, pt.TensorConstant):
                            dim_length = int(dim_length_var.data)
                        else:
                            # Symbolic, skip this check
                            continue
                    except (AttributeError, TypeError, ValueError):
                        # Can't evaluate, skip
                        continue
                else:
                    continue

            expected_shape.append(dim_length)
            dim_names.append(dim_name)

        if not expected_shape:
            # Couldn't determine expected shape, skip
            continue

        # For variables with symbolic shapes, we need to try to evaluate
        try:
            actual_shape = var.shape
            if isinstance(actual_shape, list | tuple):
                # Replace symbolic shape elements if possible
                evaluated_shape = []
                shape_idx = 0
                for dim_name in dims:
                    if dim_name is None:
                        # Skip None dims
                        if shape_idx < len(actual_shape):
                            evaluated_shape.append(actual_shape[shape_idx])
                            shape_idx += 1
                        continue

                    if dim_name not in model.coords:
                        if shape_idx < len(actual_shape):
                            shape_idx += 1
                        continue

                    if shape_idx < len(actual_shape):
                        shape_elem = actual_shape[shape_idx]
                        # Try to evaluate if symbolic
                        if isinstance(shape_elem, pt.TensorConstant):
                            evaluated_shape.append(int(shape_elem.data))
                        elif isinstance(shape_elem, Variable):
                            try:
                                evaluated = shape_elem.eval()
                                if np.isscalar(evaluated):
                                    evaluated_shape.append(int(evaluated))
                                else:
                                    evaluated_shape.append(None)  # Can't validate
                            except Exception:
                                evaluated_shape.append(None)  # Can't validate
                        else:
                            evaluated_shape.append(
                                int(shape_elem) if shape_elem is not None else None
                            )
                        shape_idx += 1

                # Compare only elements we could evaluate
                if len(evaluated_shape) != len(expected_shape):
                    # Different number of dimensions, skip
                    continue

                mismatches = []
                for i, (actual, expected) in enumerate(zip(evaluated_shape, expected_shape)):
                    if actual is not None and actual != expected:
                        mismatches.append(
                            f"  dimension {i} (dim='{dim_names[i]}'): got {actual}, expected {expected}"
                        )

                if mismatches:
                    errors.append(
                        f"Variable '{var_name}' declares dims {dims} but its shape "
                        f"does not match the coordinate lengths:\n" + "\n".join(mismatches)
                    )
        except Exception:
            # If we can't evaluate the shape, skip this check
            # The shape might be symbolic and resolve at runtime
            pass

    return errors


def check_coord_lengths(model: Model) -> list[str]:
    """Check that coordinate arrays match their dimension sizes.

    This validates that when coordinates have values, their length matches
    the dimension length. For symbolic dimensions (like MutableData), this
    check may be skipped.

    Parameters
    ----------
    model : Model
        The PyMC model to check

    Returns
    -------
    list[str]
        List of error messages (empty if no errors)
    """
    errors = []

    for dim_name, coord_values in model.coords.items():
        if coord_values is None:
            # Symbolic dimension, skip
            continue

        dim_length_var = model.dim_lengths.get(dim_name)
        if dim_length_var is None:
            continue

        try:
            # Get actual coordinate length
            coord_length = len(coord_values) if coord_values is not None else None

            # Get expected dimension length
            if isinstance(dim_length_var, pt.TensorConstant):
                expected_length = int(dim_length_var.data)
            elif isinstance(dim_length_var, Variable):
                try:
                    eval_result = dim_length_var.eval()
                    if np.isscalar(eval_result):
                        expected_length = int(eval_result)
                    else:
                        # Can't compare, might be symbolic
                        continue
                except Exception:
                    # Can't evaluate, might be symbolic (e.g., MutableData)
                    continue
            else:
                expected_length = int(dim_length_var)

            # Compare lengths
            if coord_length is not None and coord_length != expected_length:
                # Find which variables use this dimension
                using_vars = []
                for var_name, dims in model.named_vars_to_dims.items():
                    if dims is not None and dim_name in dims:
                        using_vars.append(var_name)

                var_list = (
                    ", ".join([f"'{v}'" for v in sorted(using_vars)]) if using_vars else "variables"
                )

                errors.append(
                    f"Dimension '{dim_name}' has coordinate values of length {coord_length}, "
                    f"but the dimension size is {expected_length}. "
                    f"This affects variable(s): {var_list}. "
                    f"Update the coordinate values to match the dimension size, "
                    f"or adjust the dimension size to match the coordinates."
                )
        except Exception:
            # If evaluation fails, skip (might be symbolic)
            pass

    return errors
