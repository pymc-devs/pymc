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

from typing import List, Optional

import numpy as np
import math
import pytensor.tensor as pt
import pytensor

from pytensor.graph.basic import Node
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.math import Max, MaxAndArgmax

from pymc.logprob.abstract import (
    MeasurableVariable,
    _logcdf,
    _logprob,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import ignore_logprob


class MeasurableMax(MaxAndArgmax):
    """A placeholder used to specify a log-likelihood for a clipped RV sub-graph."""

MeasurableVariable.register(MeasurableMax)

@node_rewriter([MaxAndArgmax])
def find_measurable_max(fgraph: FunctionGraph, node: Node) -> Optional[List[MeasurableMax]]:
    
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    if isinstance(node.op, MeasurableMax):
        return None  # pragma: no cover

    
    max_var = node.outputs[0]
    base_var = node.inputs[0]


    if not (
        base_var.owner
        and isinstance(base_var.owner.op, MeasurableVariable)
        and base_var not in rv_map_feature.rv_values
    ):
        return None
    
    if(base_var.owner.inputs[3].type.ndim != 0):
        return None
    
    
    # Make base_var unmeasurable
    unmeasurable_base_var = ignore_logprob(base_var)
    axis = node.op.axis
    measurable_max =  MeasurableMax(list(axis))
    max_rv_node = measurable_max.make_node(unmeasurable_base_var)
    max_rv = max_rv_node.outputs

    max_rv[0].name = max_var.name
    max_rv[1].name = node.outputs[1].name

    return max_rv


measurable_ir_rewrites_db.register(
    "find_measurable_max",
    find_measurable_max,
    "basic",
    "max",
)


@_logprob.register(MeasurableMax)
def max_logprob(op, values, base_rv, **kwargs):
   
    (value,) = values

    base_rv_op = base_rv.owner.op
    base_rv_inputs = base_rv.owner.inputs

    logprob = _logprob(base_rv_op, (value,), *base_rv_inputs, **kwargs)
    logcdf = _logcdf(base_rv_op, value, *base_rv_inputs, **kwargs)

    if base_rv_op.name:
        logprob.name = f"{base_rv_op}_logprob"
        logcdf.name = f"{base_rv_op}_logcdf"

    size_var = base_rv.owner.inputs[1]
    string_size = str(size_var)
    for b in (0, len(string_size)-1):
        if(string_size[b] == '}'):
            a = string_size[b-1]
    try :
        n = int(a)
    except ValueError:
        return None

    logprob = (n-1)*logcdf + logprob

    return logprob
