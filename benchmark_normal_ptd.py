#   Copyright 2026 - present The PyMC Developers
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
"""One-off benchmark for pymc-devs/pymc#8376 (to be removed before merge).

Compares the compiled ``model.logp_dlogp_function(ravel_inputs=True)`` graph and
its evaluation time before and after delegating Normal's logp/logcdf/logccdf/icdf
to pytensor-distributions, as requested in review.

Usage (run twice, then compare)::

    python benchmark_normal_ptd.py run --label after      # on the PR branch
    git switch --detach $(git merge-base upstream/main HEAD)
    python benchmark_normal_ptd.py run --label before     # on main (old impl)
    git switch normal-pytensor-distributions
    python benchmark_normal_ptd.py compare

Outputs go to ``ptd_benchmark_results/<label>/``.
"""
# ruff: noqa: T201

import argparse
import difflib
import json
import platform
import subprocess
import timeit

from collections import Counter
from pathlib import Path

import numpy as np
import pytensor
import pytensor.tensor as pt

import pymc as pm

from pymc.blocking import DictToArrayBijection

RESULTS_DIR = Path(__file__).parent / "ptd_benchmark_results"
SEED = 20260807
TIMEIT_REPEAT = 30
TIMEIT_NUMBER = 200


def build_models() -> dict[str, pm.Model]:
    """Models exercising every Normal code path that reaches a logp graph.

    ``normal`` only uses Normal.logp; ``normal_censored`` additionally pulls
    Normal.logcdf and logccdf into the logp graph, isolating regressions to
    the specific measure function responsible.
    """
    rng = np.random.default_rng(SEED)
    data = rng.normal(0.5, 1.2, size=1_000)
    cens_data = np.clip(rng.normal(0.0, 1.0, size=500), -1.0, 1.0)

    with pm.Model() as m_basic:
        mu = pm.Normal("mu", 0.0, 10.0)  # Normal.logp (scalar prior)
        sigma = pm.HalfNormal("sigma", 5.0)  # untouched by the PR
        pm.Normal("obs", mu, sigma, observed=data)  # Normal.logp (vector likelihood)

    with pm.Model() as m_cens:
        mu = pm.Normal("mu", 0.0, 10.0)
        sigma = pm.HalfNormal("sigma", 5.0)
        pm.Normal("obs", mu, sigma, observed=data)
        pm.Censored(  # Normal.logcdf + logccdf
            "cens",
            pm.Normal.dist(mu, sigma),
            lower=-1.0,
            upper=1.0,
            observed=cens_data,
        )
    return {"normal": m_basic, "normal_censored": m_cens}


def time_call(fn, *args) -> dict:
    times = timeit.repeat(lambda: fn(*args), repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER)
    per_call_us = [t / TIMEIT_NUMBER * 1e6 for t in times]
    return {
        "min_us": min(per_call_us),
        "median_us": float(np.median(per_call_us)),
        "repeat": TIMEIT_REPEAT,
        "number": TIMEIT_NUMBER,
    }


def op_counts(fgraph) -> dict:
    return dict(Counter(type(node.op).__name__ for node in fgraph.apply_nodes).most_common())


def run(label: str) -> None:
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)

    values, timings, ops = {}, {}, {}
    for name, m in build_models().items():
        f = m.logp_dlogp_function(ravel_inputs=True)
        f.set_extra_values({})
        x = DictToArrayBijection.map(m.initial_point()).data

        with open(out / f"{name}_logp_dlogp_dprint.txt", "w") as fh:
            pytensor.dprint(f._pytensor_function, print_memory_map=True, file=fh)

        logp, dlogp = f(x)
        values[name] = {"logp": repr(float(logp)), "dlogp": [repr(float(g)) for g in dlogp]}
        timings[name] = time_call(f, x)
        ops[name] = op_counts(f._pytensor_function.maker.fgraph)

    # icdf is changed by the PR but does not appear in logp graphs; time it separately.
    q = pt.dvector("q")
    icdf_fn = pytensor.function([q], pm.icdf(pm.Normal.dist(1.5, 2.5), q))
    with open(out / "icdf_dprint.txt", "w") as fh:
        pytensor.dprint(icdf_fn, print_memory_map=True, file=fh)
    q_vals = np.linspace(0.001, 0.999, 1_000)
    values["icdf"] = [repr(float(v)) for v in icdf_fn(q_vals)[::100]]
    timings["icdf"] = time_call(icdf_fn, q_vals)
    ops["icdf"] = op_counts(icdf_fn.maker.fgraph)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    try:
        import pytensor_distributions

        ptd_version = pytensor_distributions.__version__
    except ImportError:
        ptd_version = None
    meta = {
        "label": label,
        "commit": commit,
        "pymc": pm.__version__,
        "pytensor": pytensor.__version__,
        "pytensor_distributions": ptd_version,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "floatX": pytensor.config.floatX,
    }

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "timings.json").write_text(json.dumps(timings, indent=2))
    (out / "values.json").write_text(json.dumps(values, indent=2))
    (out / "op_counts.json").write_text(json.dumps(ops, indent=2))
    print(f"[{label}] commit={commit}")
    print(json.dumps(timings, indent=2))


def compare() -> int:
    before, after = RESULTS_DIR / "before", RESULTS_DIR / "after"
    status = 0

    for name in (
        "normal_logp_dlogp_dprint.txt",
        "normal_censored_logp_dlogp_dprint.txt",
        "icdf_dprint.txt",
    ):
        a, b = (before / name).read_text(), (after / name).read_text()
        if a == b:
            print(f"GRAPH {name}: IDENTICAL ({len(b.splitlines())} lines)")
        else:
            status = 1
            diff = list(
                difflib.unified_diff(a.splitlines(True), b.splitlines(True), "before", "after")
            )
            print(f"GRAPH {name}: DIFFERS ({len(diff)} diff lines, first 40 shown)")
            print("".join(diff[:40]))

    va = json.loads((before / "values.json").read_text())
    vb = json.loads((after / "values.json").read_text())
    if va == vb:
        print("VALUES: logp/dlogp/icdf bit-identical across runs")
    else:
        status = 1
        print(f"VALUES DIFFER:\n  before: {va}\n  after:  {vb}")

    ta = json.loads((before / "timings.json").read_text())
    tb = json.loads((after / "timings.json").read_text())
    print(
        f"\n{'bench':<12} {'before min':>11} {'after min':>11} {'ratio':>6}   "
        f"{'before med':>11} {'after med':>11}"
    )
    for key in ta:
        r = tb[key]["min_us"] / ta[key]["min_us"]
        print(
            f"{key:<12} {ta[key]['min_us']:>9.1f}us {tb[key]['min_us']:>9.1f}us {r:>6.3f}   "
            f"{ta[key]['median_us']:>9.1f}us {tb[key]['median_us']:>9.1f}us"
        )

    oa = json.loads((before / "op_counts.json").read_text())
    ob = json.loads((after / "op_counts.json").read_text())
    for key in oa:
        if oa[key] != ob[key]:
            print(f"\nOP COUNTS {key} changed:")
            for op in sorted(set(oa[key]) | set(ob[key])):
                na, nb = oa[key].get(op, 0), ob[key].get(op, 0)
                if na != nb:
                    print(f"  {op}: {na} -> {nb}")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--label", choices=["before", "after"], required=True)
    sub.add_parser("compare")
    cli_args = parser.parse_args()
    if cli_args.cmd == "run":
        run(cli_args.label)
    else:
        raise SystemExit(compare())
