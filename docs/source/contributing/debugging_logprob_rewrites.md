# Debugging Logprob Rewrites

This guide is for contributors working on `pymc.logprob` rewrite issues.
It focuses on fast, reproducible debugging loops when a measurable rewrite
is unexpectedly applied (or not applied).

## Mental model

The logprob pipeline builds a measurable IR graph from valued random variables:

1. Wrap valued RVs with `ValuedRV` nodes.
2. Apply IR rewrites (`construct_ir_fgraph`).
3. Derive logprob from measurable nodes.
4. Run cleanup rewrites.

Main entry points:

- `pymc.logprob.rewriting.construct_ir_fgraph`
- `pymc.logprob.basic.conditional_logp`
- `pymc.logprob.basic.logp`

## Minimal debugging workflow

1. Build a tiny graph with one suspected rewrite pattern.
2. Call `construct_ir_fgraph` directly.
3. Count/inspect measurable ops in the IR.
4. Add a focused regression test.
5. Fix rewrite guard conditions and re-run targeted tests.

Example snippet:

```python
import pytensor.tensor as pt

from pymc.logprob.abstract import MeasurableOp
from pymc.logprob.rewriting import construct_ir_fgraph

x_rv = pt.random.normal()
y_rv = pt.clip(x_rv, 0, 1)
z_rv = pt.random.normal(y_rv, 1, name="z")
z_vv = z_rv.clone()

fg = construct_ir_fgraph({z_rv: z_vv})
print(sum(isinstance(node.op, MeasurableOp) for node in fg.apply_nodes))
```

## What to inspect in rewrite bugs

- `node.inputs` and `node.outputs` around rewritten nodes.
- Client chains in `fgraph.clients` (especially whether a path reaches `ValuedRV`).
- Whether a rewrite crossed an RV boundary and changed graph semantics.
- Whether rewrite ordering caused a rule to fire too early.

Useful helpers:

- `pymc.logprob.utils.filter_measurable_variables`
- `pymc.logprob.utils.check_potential_measurability`
- `pymc.logprob.utils.get_related_valued_nodes`

## Testing strategy

Prefer small tests in `tests/logprob/` that assert structural properties,
for example:

- expected number of measurable ops
- expected op class presence/absence
- stable numeric logp value for a hand-checkable case

Targeted test runs:

```bash
python -m pytest tests/logprob/test_composite_logprob.py::test_unvalued_ir_reversion -q
python -m pytest tests/logprob/test_transforms.py -q
```

If your local environment cannot compile C extensions, run targeted tests with:

```bash
PYTENSOR_FLAGS=cxx= python -m pytest <target> -q
```

## Common pitfalls

- Adding broad rewrite rules without valued-path checks.
- Assuming any measurable input implies the output should also be measurable.
- Ignoring RV boundaries in client traversal.
- Fixing a local case but not adding a regression test.
