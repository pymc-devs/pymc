(pytensor_compilation)=
# PyTensor, WTF?

**Author:** [Benjamin Vincent](https://github.com/drbenvincent)

In your day-to-day modeling work, it is possible to write models exclusively in PyMC and never know about, or write any, PyTensor code. But PyTensor is a _crucial_ part of the modeling stack, and does a lot of work on your behalf every time you sample. Understanding what it is and how it works can help  explain two things PyMC users often wonder about:

- **Why does sampling sometimes take a while to even start?** That's compilation.
- **Why is the first run slow but later runs faster?** That's caching.

This guide gives you a mental model of the pipeline and the choices available, without assuming you've ever written a line of PyTensor.

## The one-sentence mental model

When PyMC samples your model, it asks PyTensor to turn the model's math (the log-probability and its gradient) into a fast, callable function. 

$$
\theta \to \big(\log(p(\theta)), \nabla \log(p(\theta))\big)
$$

PyTensor does this in two stages: it first rewrites the symbolic math into a better form, then links that form into something executable using a chosen backend.



## The pipeline at a glance

:::{figure} ../../images/pytensor_compilation.svg
:width: 100%
:alt: A flowchart of the PyTensor compilation pipeline, from the PyMC model through graph preparation, Stage 1 graph rewriting, backend selection, and Stage 2 linking, to the compiled callable consumed by an MCMC sampler engine.

The full pipeline. Everything from the PyMC model down to the **compiled callable** is PyTensor's job; everything below it is the *sampler engine*. The two blue diamonds are the two choices you control: the **backend** (how the math is compiled) and the **sampler engine** (what repeatedly evaluates it).
:::

## Walking through the stages

### Graph preparation

PyMC hands PyTensor a symbolic graph — a recipe of mathematical operations, not yet any numbers. PyTensor clones it, resolves shared variables and updates, and wraps it in a {class}`~pytensor.graph.fg.FunctionGraph`. This is bookkeeping; it's cheap.

### Stage 1 — rewriting (the "optimizer")

Think of the graph PyMC hands over as a **rough draft** of your model's math. Before running anything, PyTensor behaves like a careful editor working through that draft: it rewrites the math to run faster, to use less memory, and — especially important for probabilistic models — to be numerically safe. The key thing is that none of these edits change *what* the math computes. They only change *how* it's written down. The answer is identical; the path to it is better.

Two things are worth holding onto before we look at the individual edits:

- **It all happens in plain Python**, and the bigger your model, the longer it takes. This editing pass is a large part of the "why does sampling take a while to *start*?" pause. A big hierarchical model is a big draft, and a big draft takes longer to edit.
- **It happens the same way regardless of backend.** Whether you'll eventually run on Numba, C, JAX, PyTorch, or MLX, almost all of this rewriting happens first, in exactly the same way.

The subsections below walk through each editing pass, in the order they run — matching the "Stage 1" lane in the diagram. For a full catalogue of every available rewrite, see the {ref}`graph rewrites reference <pytensor:optimizations>`.

#### The to-do list of edits

All the possible edits live in one long, **ordered to-do list**. Each edit has a number that fixes where it sits in the queue, and a few labels attached to it. When you compile, PyTensor does *not* run every edit on the list — it runs only the ones whose labels match the mode you picked. That is the real difference between `FAST_COMPILE` (run just a handful of quick edits, so you can start sampling sooner) and the default (run the full set, so sampling itself is fast). Choosing a compilation mode is really choosing *which edits are allowed to run*.

:::{dropdown} Under the hood
Every rewrite lives in `pytensor.compile.mode.optdb`, a `pytensor.graph.rewriting.db.SequenceDB`. Entries register at a numeric `position` (merge at `0`, canonicalize at `1`, stabilize at `1.5`, specialize at `2`, fusion at `49`, inplace at `49.5`+) and run in that order. Each carries *tags* like `"fast_run"`, `"fast_compile"`, `"numba"`, `"inplace"`. The {class}`~pytensor.compile.mode.Mode` pairs a linker (Stage 2) with an *optimizer query* (`pytensor.graph.rewriting.db.RewriteDatabaseQuery`) that selects rewrites by tag: the default Numba mode queries `include=["fast_run", "numba"]`, while `FAST_COMPILE` queries `include=["fast_compile", "py_only"]`.
:::

#### Tidying up: `merge` and `useless`

The first edits are easy wins. **Merge** spots when the same calculation appears more than once and makes the model compute it a single time, then reuse the result — like noticing you've added up the same column of numbers twice and deciding to do it once. **Useless** deletes steps that can't possibly change the answer: adding `0`, multiplying by `1`, or reshaping an array into the shape it already has. Small savings each, but they also leave a cleaner draft for the heavier edits that follow.

:::{dropdown} Under the hood
Merge is `pytensor.graph.rewriting.basic.MergeOptimizer` (registered as `merge1`, `merge1.1`, `merge2`, `merge3` at several positions). "Useless" is the `"useless"` entry at position `0.6`, a `pytensor.graph.rewriting.db.TopoDB` wrapping a `LocalGroupDB` of small `local_useless_*` node rewriters.
:::

#### Speaking with one voice: `canonicalize`

There are usually many equally-valid ways to write the same expression. Left alone, that variety would force every later edit to recognise dozens of slightly different spellings of the same thing. **Canonicalize** prevents that by rewriting everything into one agreed-upon *standard form* — a house style for the math. Once the whole draft is written consistently, the later passes only have to look for one version of each pattern. This pass runs in a loop, applying its edits over and over until the draft stops changing (mathematicians call that reaching a *fixed point*); it keeps tidying until there's nothing left to tidy.

:::{dropdown} Under the hood
Canonicalize is an `pytensor.graph.rewriting.db.EquilibriumDB` at position `1` — "equilibrium" being the run-until-nothing-changes loop. Developers add a rewrite to it with the `@register_canonicalize` decorator from `pytensor.tensor.rewriting.basic`. Typical work: flattening `add`/`mul` chains, folding constants, and moving operations into a canonical position.
:::

#### Keeping the numbers safe: `stabilize`

This is the pass PyMC users benefit from the most, even though they almost never see it work. Here is the problem it solves. Computers store numbers with only so many digits of precision, and some formulas that are perfectly correct on paper fall apart on a real computer.

Take `log(1 + x)` when `x` is tiny — say `0.000000001`. The computer first works out `1 + x`, but it can't keep all those digits, so it rounds the result to exactly `1`. Then `log(1) = 0`, and your small-but-real value has simply vanished. The mathematically-identical function `log1p(x)` computes the same thing *without ever forming `1 + x`*, so the precision survives. **Stabilize** automatically swaps these fragile formulas for safe equivalents like that one.

:::{tip}
If you've ever wondered why PyMC so rarely spits out `NaN` or `inf` from ordinary-looking expressions full of logs and exponentials, this pass is a big part of the answer. It is quietly rewriting the danger out of your log-probability.
:::

:::{dropdown} Under the hood
Stabilize is an `EquilibriumDB` at position `1.5`, populated via `@register_stabilize`. Concrete examples from `pytensor.tensor.rewriting.math`: `local_log1p` (`log(1 + x)` → `log1p(x)`), `local_expm1` (`exp(x) - 1` → `expm1(x)`), and `local_log1p_plusminus_exp` (`log1p(exp(x))` → `log1pexp(x)`, the softplus). These avoid the cancellation and overflow the naive forms suffer at the extremes.
:::

#### Shortcuts and fewer passes over the data: `specialize` and `fusion`

Two more speed-ups, both about doing less work. **Specialize** swaps a general-purpose operation for a faster special-case version whenever one exists — the way you'd compute `x²` as `x * x` instead of starting up a general "raise to any power" routine.

**Fusion** fixes a subtler waste. Imagine your model computes `a * b + c` element by element across big arrays. Done naively, the computer builds one whole temporary array to hold `a * b`, then another to hold `+ c` — a lot of memory shuffled around for nothing. Fusion merges these element-wise steps so each element gets *all* of its operations in a single pass, with no temporaries in between. Picture an assembly line where each item is fully finished at one station, instead of the entire batch being carried back and forth between stations.

:::{dropdown} Under the hood
Specialize is an `EquilibriumDB` at position `2` (`@register_specialize`). Fusion is the `"elemwise_fusion"` sequence — the `FusionOptimizer`, which builds a single `Composite` op — plus `"add_mul_fusion"`, both in `pytensor.tensor.rewriting.elemwise`. Fusion is tagged `"fusion"`, which is why `FAST_COMPILE` (and JAX, which fuses on its own) leave it out.
:::

#### Last-minute, backend-aware edits: backend-specific rewrites and in-place ops

The final edits depend on *where* your model is headed and on squeezing out memory. **Backend-specific** edits are small tweaks tailored to the backend you chose — a few extra tricks that only make sense for Numba, or only for JAX, and so on. Tricks that only help the C backend are dropped when you're not using it.

**In-place** edits save memory by letting an operation write its result directly over its input's memory, instead of allocating a brand-new array — like reusing the same sheet of scratch paper rather than grabbing a fresh one every time. Overwriting is risky, though: what if some other part of the graph still needs that data? So before any in-place edit runs, a "destroy handler" acts as a safety inspector, tracking exactly which chunks of memory are safe to reuse. (JAX skips in-place edits altogether — it manages memory in its own way.)

:::{dropdown} Under the hood
Backend-specific rewrites are pulled in by the mode's tag query (`pytensor.tensor.rewriting.numba`, `pytensor.tensor.rewriting.jax`, …); C-only and BLAS rewrites are excluded for the array-library backends. In-place rewrites are tagged `"inplace"`: `AddDestroyHandler` (position `49.5`) attaches a `pytensor.graph.destroyhandler.DestroyHandler` that tracks clobberable buffers, then rewrites like `InplaceElemwiseOptimizer` (`pytensor.tensor.rewriting.elemwise`, position `50.5`) switch ops to their in-place variants.
:::

#### The finished draft

The result of all this editing is a new graph: the same inputs and outputs as before, but leaner, numerically safer, and tailored to your backend. Nothing has actually *run* yet — we've only improved the recipe. Turning that recipe into a program the computer can execute is Stage 2's job.

:::{dropdown} Under the hood
The output is a rewritten {class}`~pytensor.graph.fg.FunctionGraph`. To see it without compiling a full function, apply the rewrites directly with `pytensor.graph.rewriting.utils.rewrite_graph(y, include=("canonicalize", "specialize"))` and print the result with `pytensor.dprint`.
:::

### Stage 2 — linking (choosing a backend)

After Stage 1 you have a polished recipe — but a recipe is just instructions, not a meal. **Linking** is the step that turns those instructions into a real program the computer can run quickly. The recipe is still written in PyTensor's own internal language; linking translates it into a language that actually executes, and (for most backends) compiles it. *How* that translation happens, and how fast the result runs, depends on the **backend** you chose.

Here's the lay of the land before we dig in:

| Backend | How it turns the recipe into a program | Needs a system compiler? | Good to know |
|---|---|---|---|
| **Numba** *(default)* | Converts the recipe to Python, then compiles it to fast machine code the first time you run it (via LLVM) | No (ships ready-to-use) | Why `pip install pymc` "just works" (from PyMC 6.0 onwards) without conda. First run is slow; results are cached on disk. |
| **C / CVM** | Writes out C++ source and compiles it with `g++`/`clang++` | Yes | Historically the default (pre PyMC 6.0); very strong on-disk cache. Now opt-in. |
| **JAX** | Converts to JAX, compiles via `jax.jit` → XLA | No | The path to GPU/TPU. |
| **PyTorch** | Converts to PyTorch, compiles via `torch.compile` | No | Runs on PyTorch's CPU/GPU devices. |
| **MLX** | Converts to MLX, compiles via `mx.compile` | No | Targets the GPU in Apple Silicon Macs. |
| **Python VM** | Just runs each operation in plain Python — no compiling | No | Slowest to run, instant to "build". Great for debugging. |

The default backend is **Numba** (it's what `config.linker = "auto"` resolves to). For a deeper treatment of how modes and linkers work, see {ref}`modes and linkers <pytensor:using_modes>`.

#### How the backend gets chosen

Just as `nuts_sampler=` picks the sampler engine, a different knob picks the PyTensor backend. From PyMC, the simplest way is the `backend=` argument to `pm.sample`:

```python
import pymc as pm

with model:
    # Default-ish: Numba. Other common choices: "c", "jax".
    idata = pm.sample(backend="numba")
```

Under the hood, PyMC turns `backend=` into a PyTensor *compilation mode* (note that `backend="c"` maps to PyTensor's combined C+VM mode, `"cvm"`). If you need finer control you can pass a mode directly via `compile_kwargs` instead — but you can only set one of the two:

```python
with model:
    # Equivalent to backend="jax"; do NOT also pass backend=...
    idata = pm.sample(compile_kwargs={"mode": "JAX"})
```

If you're calling PyTensor directly (no PyMC), the same choice is made with `config.linker` for the whole session, or per call via the `mode` argument to `pytensor.function`:

```python
import pytensor

# Session-wide default for every compiled function
pytensor.config.linker = "jax"

# Or per function, overriding the session default
f = pytensor.function([x], y, mode="NUMBA")
```

The two choices — backend and sampler — are made at different points in the pipeline (see the two blue decision diamonds in the diagram), and they're mostly independent. You pick the engine with `pm.sample(nuts_sampler=...)`:

- **`"pymc"`** — PyMC's built-in NUTS. The loop is pure Python and calls a PyTensor callable from any CPU backend (typically C or Numba).
- **`"nutpie"`** — runs the NUTS loop in Rust (via [`nuts-rs`](https://github.com/pymc-devs/nutpie)). It calls a PyTensor callable compiled to either the **Numba** or the **JAX** backend. This is the default when nutpie is installed.
- **`"numpyro"`** — NumPyro's NUTS, whose loop runs in **JAX**. It requires the **JAX** backend.
- **`"blackjax"`** — BlackJAX's NUTS, also a **JAX** loop requiring the **JAX** backend.

The one coupling to be aware of: the JAX-based engines (`numpyro`, `blackjax`, and `nutpie` in JAX mode) need PyTensor to have compiled the callable to JAX. PyMC handles wiring this up for you.

#### The one idea behind (almost) every backend

Most backends work the same way under the hood, and once you see it the rest is just details. PyTensor keeps a kind of **phrasebook**: for every operation in your graph, it knows how to say that operation in the target language. To link a whole graph, it translates each operation in turn, stitches the translations together into one program, and then lets the backend compile that program. Adding support for a new operation in a backend is just adding one new phrase to that backend's phrasebook.

The C and Python backends are the two exceptions — instead of a phrasebook they ask each operation directly for its hand-written C code, or simply run the operation's plain-Python implementation. The rest of this section takes the backends one at a time (the branches of the diagram's "Stage 2" lane).

:::{dropdown} Under the hood
Each backend is a *linker* (a `pytensor.link.basic.Linker` subclass); the {class}`~pytensor.compile.mode.Mode`'s linker is what Stage 2 runs. The "phrasebook" is a single-dispatch registry: `pytensor.link.numba.dispatch.numba_funcify`, `pytensor.link.jax.dispatch.jax_funcify`, `pytensor.link.pytorch.dispatch.pytorch_funcify`, `pytensor.link.mlx.dispatch.mlx_funcify`. The C and Python backends instead use `Op.c_code` and `Op.perform`.
:::

#### Numba (the default)

Numba turns your recipe into Python code, then hands it to a **just-in-time (JIT) compiler** that converts it into fast machine code the very first time you run it. That's why the first call is slow — it includes the compiling — while every call afterwards is quick. Better still, the compiled result is saved to disk, so even a brand-new Python session can usually skip the wait.

Here's the intuition: the first time you cook an unfamiliar dish you're slow, reading each line and hunting for ingredients. By the tenth time it's second nature. Numba writes down that "second nature" version so it never has to relearn it. And because Numba comes ready-to-use when you `pip install`, there's no compiler for you to set up — which is exactly why `pip install pymc` "just works".

:::{dropdown} Under the hood
`pytensor.link.numba.linker.NumbaLinker`. `numba_funcify` emits one Python function wiring together each op's Numba implementation; `pytensor.link.numba.dispatch.numba_njit` JIT-compiles it with [`numba.njit`](https://numba.readthedocs.io/en/stable/user/jit.html) via LLVM. With `config.numba__cache = True` (default) the result is cached on disk (see `numba_funcify_and_cache_key`).
:::

#### C / CVM

The C backend writes out C++ source code and compiles it using a real C++ compiler (`g++` or `clang++`) installed on your machine. It runs fast and keeps an excellent on-disk cache — but it only works if those build tools are present. That requirement is the whole reason it's no longer the default: plenty of people who `pip install pymc` don't have a C++ compiler set up. The everyday `"cvm"` variant is a practical hybrid — a small, fast loop written in C marches through your operations, each of which is either compiled C or falls back to Python.

:::{dropdown} Under the hood
`pytensor.link.c.basic.CLinker` generates C++ from each op's `Op.c_code`, compiles a `.so`, and caches it in your compile directory. `"cvm"` is `pytensor.link.vm.VMLinker` with `use_cloop=True` (the C loop steps through op *thunks*); pure `"c"` is `CLinker` / `OpWiseCLinker`.
:::

#### JAX

JAX translates the recipe into JAX code and lets JAX's own compiler (XLA) take over. This is the route to **GPUs and TPUs**, and it's the backend the JAX-based samplers (`numpyro`, `blackjax`, and `nutpie` in its JAX mode) plug straight into.

:::{dropdown} Under the hood
`pytensor.link.jax.linker.JAXLinker`. `jax_funcify` builds a JAX-traceable function that the linker hands to [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) → XLA.
:::

#### PyTorch

Same idea, using PyTorch instead. The recipe is translated into PyTorch and compiled with [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), running on whatever CPU or GPU device PyTorch is set up to use. Handy when the rest of your stack already lives in PyTorch.

:::{dropdown} Under the hood
`pytensor.link.pytorch.linker.PytorchLinker`; `pytorch_funcify` builds the function, compiled via `torch.compile` (TorchDynamo + Inductor).
:::

#### MLX

The same idea once more, this time aimed at the GPU built into Apple Silicon Macs.

:::{dropdown} Under the hood
`pytensor.link.mlx.linker.MLXLinker`; `mlx_funcify` builds an MLX function compiled lazily with `mx.compile`.
:::

#### Python VM

The simplest backend of all: **no compiling whatsoever**. PyTensor just runs each operation one at a time in plain Python. That makes it the slowest to actually run, but instant to "build" and by far the easiest to debug (you can drop into any step with ordinary Python tools). It's what `FAST_COMPILE` uses, and it's a great companion while you're still iterating on a model's structure.

:::{dropdown} Under the hood
`pytensor.link.basic.PerformLinker` (or `pytensor.link.vm.VMLinker` with `use_cloop=False`) steps through the graph calling each op's `Op.perform`.
:::

### Runtime

Functionally, the compiled callable is a map from a **point in parameter space** to the model's **log-probability and its gradient**: `θ → (logp(θ), ∇logp(θ))`. The observed data isn't an argument — it's baked into the function as constants when the function is built. (PyMC actually compiles a few such functions: the log-density and its gradient for sampling, plus functions for drawing from priors/posterior predictive.)

An MCMC sampler calls this function many times, proposing new parameter points and reading back `logp` and its gradient to decide where to go next. Time spent *here* is sampling time, which is separate from the compilation time described above — speeding up compilation does not speed up sampling, and vice versa.

**This is the key boundary.** Everything above the compiled callable in the diagram is *PyTensor* turning your model's math into a function. Everything below it is a *sampler engine* repeatedly calling that function. The sampler is a separate piece of software with its own loop; PyTensor's job is finished once the callable exists.

A common misconception is worth heading off: with nutpie it's easy to assume "the fast Rust thing" is doing the compilation. It isn't — the Rust is the *sampler loop*, not a PyTensor compilation backend. PyTensor still builds the logp/gradient function that the Rust loop evaluates.

## Caching: why the second run is faster

PyTensor caches the **linked artifact** so repeated builds can skip recompilation:

- **Numba** keeps an on-disk cache (enabled by default via `numba__cache`).
- **C / CVM** caches compiled `.so` modules in your compile directory.

Note a current limitation that matters in practice: the cache is for the *linked* output. Stage 1 rewriting is **not** cached and is re-run on every build, even for a structurally identical model.

## Practical tips for PyMC users

- **Iterating on model structure? Compile faster, run slower.** Use the `FAST_COMPILE` mode, which skips many rewrites and uses the pure-Python VM (no C/LLVM compilation at all). In PyMC you can often pass this via `compile_kwargs={"mode": "FAST_COMPILE"}`. Switch back to the default for the real, long sampling run.
- **Want to know where the time goes?** Turn on profiling:

  ```python
  import pytensor
  pytensor.config.profile = True            # per-function rewrite vs link time
  pytensor.config.profile_optimizer = True  # per-rewrite breakdown
  ```

- **"It recompiles every morning."** That usually means the on-disk cache isn't being hit across sessions. Check that your compile directory is stable and not being cleared between runs.
- **Choosing a backend.** Stick with the default (Numba) unless you have a specific reason: JAX for GPU/TPU, PyTorch to integrate with the PyTorch ecosystem, MLX for Apple Silicon GPUs, C/CVM if you specifically need the C runtime characteristics.

## Where to go next

- {ref}`pymc_pytensor` — how PyMC models become PyTensor graphs in the first place.
- {ref}`graph rewrites catalogue <pytensor:optimizations>` — the rewrites applied in Stage 1.
- {ref}`modes and linkers <pytensor:using_modes>` — modes and linkers in more depth.
- {doc}`Mode / linker API reference <pytensor:library/compile/mode>`.
