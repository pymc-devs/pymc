# One-off `logp_dlogp` analysis for #8376 (to be removed before merge)

Requested in review: an analysis of `model.logp_dlogp_function(ravel_inputs=True)` —
the compiled graph (`print_memory_map=True`) and a timing that confirms no
graph/eval regression after delegating Normal's `logp`/`logcdf`/`logccdf`/`icdf`
to [pytensor-distributions](https://github.com/pymc-devs/pytensor-distributions).

Produced by `benchmark_normal_ptd.py` (repo root). Raw artifacts (dprints,
timings, op counts, values) live in `ptd_benchmark_results/{before,after}/`.

## Reproduce

```bash
python benchmark_normal_ptd.py run --label after      # on this branch
git switch --detach $(git merge-base upstream/main HEAD)
python benchmark_normal_ptd.py run --label before     # old implementation
git switch normal-pytensor-distributions
python benchmark_normal_ptd.py compare
```

## Environment

| | |
|---|---|
| before / after commits | `3b661c7e5` (merge-base with main) / `83cb13865` (this PR) |
| pytensor / pytensor-distributions | 3.2.2 / 0.2.0 |
| numpy / python | 2.4.6 / 3.13.11 |
| machine / floatX | macOS 26.5.2 arm64 (M-series) / float64 |

## Benchmarks

1. **`normal`** — `mu ~ Normal`, `sigma ~ HalfNormal`, `obs ~ Normal(mu, sigma)` with
   N=1000 observed. Exercises `Normal.logp` (scalar prior + vector likelihood).
2. **`normal_censored`** — same plus `Censored(Normal.dist(mu, sigma), -1, 1)` with
   N=500 observed, which pulls `Normal.logcdf` **and** `Normal.logccdf` into the
   logp graph.
3. **`icdf`** — standalone compiled `pm.icdf(pm.Normal.dist(1.5, 2.5), q)` (not part
   of any logp graph, but also changed by this PR).

Timings: `timeit.repeat(repeat=30, number=200)`, best (min) per call.

## Results

| bench | graph | before min | after min | ratio | values |
|---|---|---:|---:|---:|---|
| `normal` logp_dlogp | **byte-identical dprint** (141 lines) | 2.14 µs | 2.17 µs | **1.02** (noise) | bit-identical |
| `normal_censored` logp_dlogp | differs (see below) | 26.9 µs | 36.1 µs | **1.34** | logp bit-identical; dlogp[0] differs by 1 ulp |
| `icdf` | differs by design (see below) | 7.9 µs | 8.8 µs | 1.12 | ≤ 2 ulp |

### `normal`: no regression ✅

The delegated `logpdf` is the same expression as the old inline formula, so after
rewrites the compiled `logp_dlogp` graph is **byte-for-byte identical** and evaluation
time is unchanged. This is the core code path for the vast majority of models.

### `normal_censored`: graph + eval regression from `logsf` ⚠️

`pytensor_distributions.normal` implements

```python
def logsf(x, mu, sigma):
    return logcdf(-x, -mu, sigma)
```

which is mathematically identical to the old `normal_lccdf`, but graph-wise it is
not: `-x` bakes a **negated copy of the observed data into the graph as a second
constant** (visible in the dprint as `[-5.244064 ... 58192e-01]`, duplicating the
censored data's memory), and the standardized value inside becomes
`(-x - (-mu))/sigma`, which the rewriter does not recombine into `-((x - mu)/sigma)`
— so `logcdf` and `logccdf` no longer share the `z = (x - mu)/sigma` subexpression.
Consequences on this model:

- `DimShuffle` count 8 → 12; the censored fused elemwise grows from 5 to 9
  `reduce[add]` outputs; dprint grows 500 → 571 lines.
- eval time 26.9 µs → 36.1 µs (**1.34×**) for N=500 censored points.
- `dlogp` wrt `mu` differs in the last ulp (different but equivalent expression
  arrangement); logp itself is bit-identical.

The old PyMC `normal_lccdf` applies the erfcx switch directly to `z`
(`switch(z > 1, log(erfcx(z/√2)/2) − z²/2, log1p(−erfc(−z/√2)/2))`). Writing
`logsf` that way in pytensor-distributions would restore subexpression sharing and
remove the duplicated constant — a small upstream fix; alternatively PyMC can keep
its own `logccdf` until then.

### `icdf`: intentional behavior difference, minor micro-overhead

- Old: `check_icdf_value` wraps `−erfcinv(2q)·σ√2 + μ` in a single
  `switch(0 ≤ q ≤ 1, ..., nan)`.
- New: the package's `ppf` uses `erfinv(2q − 1)` inside `ppf_bounds_cont`, which
  adds explicit branches `q<0 or q>1 → nan`, `q=0 → −inf`, `q=1 → +inf`. Same
  results (boundary infinities previously arose from `erfcinv` itself), values
  match to ≤ 2 ulp (`erfinv` vs `erfcinv`), and the extra scalar switches cost
  ~0.9 µs on a 1000-point batch. Not part of logp graphs; irrelevant for sampling.

## Compiled graph (`print_memory_map=True`), after — `normal` model

<details><summary>ptd_benchmark_results/after/normal_logp_dlogp_dprint.txt (byte-identical to before)</summary>

```text
Composite{...}.0 [id A] d={0: [1]} 8
 ├─ Subtensor{i} [id B] v={0: [0]} 1
 │  ├─ joined_inputs [id C]
 │  └─ 1 [id D]
 ├─ FusedElemwise{Composite{...}, reduce[add@(0,), add@(0,), add@(0,)]}.0 [id E] 6
 │  ├─ Composite{...}.1 [id F] 5
 │  │  └─ ExpandDims{axis=0} [id G] v={0: [0]} 4
 │  │     └─ Exp [id H] 3
 │  │        └─ Subtensor{i} [id B] v={0: [0]} 1
 │  │           └─ ···
 │  ├─ Composite{...}.0 [id F] 5
 │  │  └─ ···
 │  ├─ ExpandDims{axis=0} [id G] v={0: [0]} 4
 │  │  └─ ···
 │  ├─ obs{[ 1.320334 ... 56013e+00]} [id I]
 │  └─ Subtensor{:stop} [id J] v={0: [0]} 0
 │     ├─ joined_inputs [id C]
 │     └─ 1 [id K]
 ├─ Subtensor{i} [id L] v={0: [0]} 2
 │  ├─ joined_inputs [id C]
 │  └─ 0 [id M]
 └─ Exp [id H] 3
    └─ ···
Join{axis=0} [id N] 12
 ├─ Composite{...}.0 [id O] d={0: [0], 1: [1]} 11
 │  ├─ ExpandDims{axis=0} [id G] v={0: [0]} 4
 │  │  └─ ···
 │  ├─ ExpandDims{axis=0} [id P] v={0: [0]} 7
 │  │  └─ FusedElemwise{Composite{...}, reduce[add@(0,), add@(0,), add@(0,)]}.1 [id E] 6
 │  │     └─ ···
 │  ├─ Subtensor{:stop} [id J] v={0: [0]} 0
 │  │  └─ ···
 │  ├─ ExpandDims{axis=0} [id Q] v={0: [0]} 10
 │  │  └─ Composite{...}.1 [id A] d={0: [1]} 8
 │  │     └─ ···
 │  └─ ExpandDims{axis=0} [id R] v={0: [0]} 9
 │     └─ FusedElemwise{Composite{...}, reduce[add@(0,), add@(0,), add@(0,)]}.2 [id E] 6
 │        └─ ···
 └─ Composite{...}.1 [id O] d={0: [0], 1: [1]} 11
    └─ ···

Inner graphs:

Composite{...} [id A] d={0: [1]}
 ← add [id S]
    ├─ -3.221523658174155 [id T]
    ├─ mul [id U]
    │  ├─ -0.5 [id V]
    │  └─ sqr [id W]
    │     └─ mul [id X]
    │        ├─ 0.1 [id Y]
    │        └─ i2 [id Z]
    ├─ Switch [id BA]
    │  ├─ GE [id BB]
    │  │  ├─ i3 [id BC]
    │  │  └─ 0.0 [id BD]
    │  ├─ add [id BE]
    │  │  ├─ -1.83522929514961 [id BF]
    │  │  └─ mul [id BG]
    │  │     ├─ -0.5 [id V]
    │  │     └─ sqr [id BH]
    │  │        └─ mul [id BI]
    │  │           ├─ 0.2 [id BJ]
    │  │           └─ i3 [id BC]
    │  └─ -inf [id BK]
    ├─ i0 [id BL]
    └─ i1 [id BM]
 ← GE [id BB]
    └─ ···

FusedElemwise{Composite{...}, reduce[add@(0,), add@(0,), add@(0,)]} [id E]
 ← Sum{axes=None} [id BN]
    └─ Composite{...}.0 [id BO]
       ├─ i0 [id BP]
       ├─ i1 [id BQ]
       ├─ i2 [id BR]
       ├─ i3 [id BS]
       └─ i4 [id BT]
 ← Sum{axes=None} [id BU]
    └─ Composite{...}.1 [id BO]
       └─ ···
 ← Sum{axes=None} [id BV]
    └─ Composite{...}.2 [id BO]
       └─ ···

Composite{...} [id F]
 ← log [id BW]
    └─ i0 [id BL]
 ← GT [id BX]
    ├─ i0 [id BL]
    └─ 0 [id BY]

Composite{...} [id O] d={0: [0], 1: [1]}
 ← add [id BZ]
    ├─ mul [id CA]
    │  ├─ -0.0100000 ... 0000000002 [id CB]
    │  └─ i2 [id Z]
    └─ true_div [id CC]
       ├─ true_div [id CD]
       │  ├─ i1 [id BM]
       │  └─ i0 [id BL]
       └─ i0 [id BL]
 ← add [id CE]
    ├─ -999.0 [id CF]
    ├─ Switch [id CG]
    │  ├─ i3 [id CH]
    │  ├─ mul [id CI]
    │  │  ├─ -0.04000000000000001 [id CJ]
    │  │  ├─ i0 [id BL]
    │  │  └─ i0 [id BL]
    │  └─ 0.0 [id CK]
    └─ mul [id CL]
       ├─ true_div [id CM]
       │  ├─ i4 [id CN]
       │  └─ sqr [id CO]
       │     └─ i0 [id BL]
       └─ i0 [id BL]

Composite{...} [id BO]
 ← Switch [id CP]
    ├─ i0 [id CQ]
    ├─ sub [id CR]
    │  ├─ add [id CS]
    │  │  ├─ -0.9189385332046727 [id CT]
    │  │  └─ mul [id CU]
    │  │     ├─ -0.5 [id CV]
    │  │     └─ sqr [id CW]
    │  │        └─ true_div [id CX]
    │  │           ├─ sub [id CY]
    │  │           │  ├─ i3 [id BC]
    │  │           │  └─ i4 [id CN]
    │  │           └─ i2 [id Z]
    │  └─ i1 [id BM]
    └─ -inf [id CZ]
 ← sub [id CY]
    └─ ···
 ← mul [id DA]
    ├─ true_div [id CX]
    │  └─ ···
    └─ sub [id CY]
       └─ ···
```

</details>

Full dprints for all benchmarks (before and after) are committed under
`ptd_benchmark_results/` — `diff` the `before/` and `after/` files to see the
graph changes described above.
