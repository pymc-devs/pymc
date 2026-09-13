(pytensor_compilation)=
# PyTensor, WTF?

**Author:** [Benjamin Vincent](https://github.com/drbenvincent)

In your day-to-day modeling work, it is possible to write models exclusively in PyMC and never know about, let alone write, PyTensor code. Despite remaining in the background, the work Pytensor does on your behalf every time you sample makes it a crucial part of modeling stack. Understanding how it works helps with questions like:

- **Why does sampling sometimes take a while to even start?** That's compilation.
- **Why is the first run slow but later runs faster?** That's caching.
- **Why should I use `pm.Data` instead of passing arrays directly?** Because `pm.Data` lets you update your data without recompiling — critical for iterative workflows.
- **Why does PyMC rarely produce `NaN` or `inf` from ordinary log-probability expressions?** Numerical rewrites fix fragile formulas before anything runs.
- **Which backend should I choose, and what does that even mean?** This guide explains the options and when each matters.
- **Why does writing vectorized code matter even when the backend isn't Python?** Because vectorized expressions produce simpler graphs that the rewriter can optimise and translate more effectively.

This guide gives you a mental model of the pipeline and the choices available, without assuming you've ever written a line of PyTensor.

## The one-sentence mental model

When PyMC fits or samples a model, it asks PyTensor to turn symbolic math into fast, callable functions. What gets compiled depends on what you're doing:

- **Gradient-based MCMC** (NUTS, HMC): log-probability and its gradient — $\theta \to (\log p(\theta), \nabla \log p(\theta))$
- **Gradient-free MCMC** (Metropolis, Slice, DEMetropolis): log-probability only — $\theta \to \log p(\theta)$
- **Forward sampling** (`sample_posterior_predictive`, `sample_prior_predictive`): draws from the generative model — no log-probability at all

PyTensor does this in two stages: it first rewrites the symbolic math into a better form, then links that form into something executable using a chosen backend.



## The pipeline at a glance

:::{figure} ../../images/pytensor_compilation.svg
:width: 100%
:alt: A flowchart of the PyTensor compilation pipeline, from the PyMC model through graph preparation, Stage 1 graph rewriting, backend selection, and Stage 2 linking, to the compiled callable consumed by an MCMC sampler engine.

The full pipeline. Everything from the PyMC model down to the **compiled callable** is PyTensor's job; everything below it is the *sampler engine*. The two blue diamonds are the two choices you control: the **backend** (how the math is compiled) and the **sampler engine** (what repeatedly evaluates it).
:::

## Walking through the stages

### Graph preparation

PyMC hands PyTensor a symbolic graph — a recipe of mathematical operations, not yet any numbers. PyTensor clones it and prepares it for rewriting. This is bookkeeping; it's cheap.

### Stage 1 — rewriting (the "optimizer")

Think of the graph PyMC hands over as a **rough draft** of your model's math. Before running anything, PyTensor behaves like a careful editor working through that draft: it rewrites the math to run faster, to use less memory, and — especially important for probabilistic models — to be numerically safe. The key thing is that none of these edits change *what* the math computes. They only change *how* it's written down. The answer is identical; the path to it is better.

Three things are worth holding onto before we look at the individual edits:

- **It all happens in plain Python**, and the bigger your model, the longer it takes. This editing pass is a large part of the "why does sampling take a while to *start*?" pause. A big hierarchical model is a big draft, and a big draft takes longer to edit.
- **It happens the same way regardless of backend.** Whether you'll eventually run on Numba, C, JAX, PyTorch, or MLX, almost all of this rewriting happens first, in exactly the same way.
- **Vectorized code helps here.** PyTensor rewrites the *graph*, not Python loops. NumPy-style vectorized expressions produce a compact graph the rewriter can simplify and translate cleanly; a Python `for` loop builds a much larger, deeper graph — even though the backend that eventually runs is not Python.

The subsections below walk through each editing pass, in the order they run — matching the "Stage 1" lane in the diagram. For a full catalogue of every available rewrite, see the {ref}`graph rewrites reference <pytensor:optimizations>`.

#### The to-do list of edits

All the possible edits live in one long, **ordered to-do list**. Each edit has a number that fixes where it sits in the queue, and a few labels attached to it. When you compile, PyTensor does *not* run every edit on the list — it runs only the ones whose labels match the mode you picked. That is the real difference between `FAST_COMPILE` (run just a handful of quick edits, so you can start sampling sooner) and the default (run the full set, so sampling itself is fast). Choosing a compilation mode is really choosing *which edits are allowed to run*.

#### Tidying up: `merge`

The first edit is an easy win. **Merge** spots when the same calculation appears more than once and makes the model compute it a single time, then reuse the result — like noticing you've added up the same column of numbers twice and deciding to do it once.

#### Speaking with one voice: `canonicalize`

There are usually many equally-valid ways to write the same expression. Left alone, that variety would force every later edit to recognise dozens of slightly different spellings of the same thing. **Canonicalize** prevents that by rewriting everything into one agreed-upon *standard form* — a house style for the math. Typical tidy-ups include flattening chains of additions and multiplications, folding constants, and removing terms that cannot change the answer (adding `0`, multiplying by `1`). Once the whole draft is written consistently, the later passes only have to look for one version of each pattern. This pass runs in a loop, applying its edits over and over until the draft stops changing (reaching an *equilibrium*); it keeps tidying until there's nothing left to tidy.

#### Keeping the numbers safe: `stabilize`

This is the pass PyMC users benefit from the most, even though they almost never see it work. Here is the problem it solves. Computers store numbers with only so many digits of precision, and some formulas that are perfectly correct on paper fall apart on a real computer.

Take `log(1 + x)` when `x` is tiny — say `0.000000001`. The computer first works out `1 + x`, but it can't keep all those digits, so it rounds the result to exactly `1`. Then `log(1) = 0`, and your small-but-real value has simply vanished. The mathematically-identical function `log1p(x)` computes the same thing *without ever forming `1 + x`*, so the precision survives. **Stabilize** automatically swaps these fragile formulas for safe equivalents like that one.

:::{tip}
If you've ever wondered why PyMC so rarely spits out `NaN` or `inf` from ordinary-looking expressions full of logs and exponentials, this pass is a big part of the answer. It is quietly rewriting the danger out of your log-probability.
:::

#### Shortcuts and fewer passes over the data: `specialize` and `fusion`

Two more speed-ups, both about doing less work. **Specialize** swaps a general-purpose operation for a faster special-case version whenever one exists — the way you'd compute `x²` as `x * x` instead of starting up a general "raise to any power" routine.

**Fusion** fixes a subtler waste. Imagine your model computes `a * b + c` element by element across big arrays. Done naively, the computer builds one whole temporary array to hold `a * b`, then another to hold `+ c` — a lot of memory shuffled around for nothing. Fusion merges these element-wise steps so each element gets *all* of its operations in a single pass, with no temporaries in between. Picture an assembly line where each item is fully finished at one station, instead of the entire batch being carried back and forth between stations.

#### Last-minute, backend-aware edits: backend-specific rewrites and in-place ops

The final edits depend on *where* your model is headed and on squeezing out memory. **Backend-specific** edits are small tweaks tailored to the backend you chose — a few extra tricks that only make sense for Numba, or only for JAX, and so on. Tricks that only help the C backend are dropped when you're not using it.

**In-place** edits save memory by letting an operation write its result directly over its input's memory, instead of allocating a brand-new array — like reusing the same sheet of scratch paper rather than grabbing a fresh one every time. Overwriting is risky, though: what if some other part of the graph still needs that data? So before any in-place edit runs, a "destroy handler" acts as a safety inspector, tracking exactly which chunks of memory are safe to reuse. (JAX skips in-place edits altogether — it manages memory in its own way.)

#### The finished draft

The result of all this editing is a new graph: the same inputs and outputs as before, but leaner, numerically safer, and tailored to your backend. Nothing has actually *run* yet — we've only improved the recipe. Turning that recipe into a program the computer can execute is Stage 2's job.

### Stage 2 — linking (choosing a backend)

After Stage 1 you have a polished recipe — but a recipe is just instructions, not a meal. **Linking** is the step that turns those instructions into a real program the computer can run quickly. The recipe is still written in PyTensor's own internal language; linking translates it into a language that actually executes, and (for most backends) compiles it. *How* that translation happens, and how fast the result runs, depends on the **backend** you chose.

Most backends translate the graph operation by operation into a target language, stitch the pieces into one program, and let the backend compile it. The C and Python backends are simpler: C generates source code per operation; Python just runs each operation in sequence.

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

The two choices — backend and sampler — are made at different points in the pipeline (see the two blue decision diamonds in the diagram), and they're mostly independent. For NUTS specifically, you pick the engine with `pm.sample(nuts_sampler=...)`:

- **`"pymc"`** — PyMC's built-in NUTS. The loop is pure Python and calls a PyTensor callable from any CPU backend (typically C or Numba).
- **`"nutpie"`** — runs the NUTS loop in Rust (via [`nuts-rs`](https://github.com/pymc-devs/nutpie)). It calls a PyTensor callable compiled to either the **Numba** or the **JAX** backend. This is the default when nutpie is installed.
- **`"numpyro"`** — NumPyro's NUTS, whose loop runs in **JAX**. It requires the **JAX** backend.
- **`"blackjax"`** — BlackJAX's NUTS, also a **JAX** loop requiring the **JAX** backend.

The one coupling to be aware of: the JAX-based engines (`numpyro`, `blackjax`, and `nutpie` in JAX mode) need PyTensor to have compiled the callable to JAX. PyMC handles wiring this up for you.

#### Numba (the default)

Numba turns your recipe into Python code, then hands it to a **just-in-time (JIT) compiler** that converts it into fast machine code the very first time you run it. That's why the first call is slow — it includes the compiling — while every call afterwards is quick. Better still, the compiled result is saved to disk, so even a brand-new Python session can usually skip the wait.

Here's the intuition: the first time you cook an unfamiliar dish you're slow, reading each line and hunting for ingredients. By the tenth time it's second nature. Numba writes down that "second nature" version so it never has to relearn it. And because Numba comes ready-to-use when you `pip install`, there's no compiler for you to set up — which is exactly why `pip install pymc` "just works".

#### C / CVM

The C backend writes out C++ source code and compiles it using a real C++ compiler (`g++` or `clang++`) installed on your machine. It runs fast and keeps an excellent on-disk cache — but it only works if those build tools are present. That requirement is the whole reason it's no longer the default: plenty of people who `pip install pymc` don't have a C++ compiler set up. The everyday `"cvm"` variant is a practical hybrid — a small, fast loop written in C marches through your operations, each of which is either compiled C or falls back to Python.

#### JAX

JAX translates the recipe into JAX code and lets JAX's own compiler (XLA) take over. This is the route to **GPUs and TPUs**, and it's the backend the JAX-based samplers (`numpyro`, `blackjax`, and `nutpie` in its JAX mode) plug straight into.

#### PyTorch

Same idea, using PyTorch instead. The recipe is translated into PyTorch and compiled with [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), running on whatever CPU or GPU device PyTorch is set up to use. Handy when the rest of your stack already lives in PyTorch.

#### MLX

The same idea once more, this time aimed at the GPU built into Apple Silicon Macs.

#### Python VM

The simplest backend of all: **no compiling whatsoever**. PyTensor just runs each operation one at a time in plain Python. That makes it the slowest to actually run, but instant to "build" and by far the easiest to debug (you can drop into any step with ordinary Python tools). It's what `FAST_COMPILE` uses, and it's a great companion while you're still iterating on a model's structure.

### Runtime

What the compiled callable does depends on the task:

| Task | What PyTensor compiles | Typical samplers |
|---|---|---|
| Gradient MCMC | $\theta \to (\log p(\theta), \nabla \log p(\theta))$ | NUTS (pymc, nutpie, numpyro, blackjax), HMC |
| Gradient-free MCMC | $\theta \to \log p(\theta)$ | Metropolis, Slice, DEMetropolis |
| Forward sampling | draws from the generative model | `sample_posterior_predictive`, `sample_prior_predictive` |

PyMC may compile several such functions for one model — log-density and gradient for sampling, plus separate functions for prior or posterior predictive draws.

The way observed data is handled depends on how it was declared:

- **Plain NumPy arrays** are folded into the compiled function as constants. The compiler can see those values at build time and use them to simplify the graph — but if you want to update the data you have to recompile from scratch.
- **`pm.Data` containers** are kept as *runtime inputs* that the function reads each time it is called, not at build time. The function is compiled once; you can then call `pm.set_data(...)` to swap the values in and the compiled function simply reads the new data on its next call — no recompilation needed. The trade-off is that the compiler cannot fold those values in to simplify the graph.

This matters most in iterative workflows — posterior predictive checks over many data splits, or updating a model's observations between sampling runs — where recompiling each time would dominate wall time.

If you want data and dimension lengths treated as compile-time constants (enabling stronger graph rewrites) without restructuring your model, use the `freeze_dims_and_data` model transform, which converts `pm.Data` containers into true constants before compilation.

An MCMC sampler calls its compiled function many times, proposing new parameter points and reading back log-probability (and gradient, when needed) to decide where to go next. Time spent *here* is sampling time, which is separate from the compilation time described above — speeding up compilation does not speed up sampling, and vice versa.

**This is the key boundary.** Everything above the compiled callable in the diagram is *PyTensor* turning your model's math into a function. Everything below it is a *sampler engine* repeatedly calling that function. The sampler is a separate piece of software with its own loop; PyTensor's job is finished once the callable exists.

A common misconception is worth heading off: with nutpie it's easy to assume "the fast Rust thing" is doing the compilation. It isn't — the Rust is the *sampler loop*, not a PyTensor compilation backend. PyTensor still builds the function the Rust loop evaluates.

## Caching: why the second run is faster

PyTensor caches the **linked artifact** so repeated builds can skip recompilation:

- **Numba** keeps an on-disk cache (enabled by default via `numba__cache`).
- **C / CVM** caches compiled `.so` modules in your compile directory.

Note a current limitation that matters in practice: today, PyTensor's on-disk cache covers the *linked* output (Stage 2), but Stage 1 graph rewriting is re-run on every new compilation of a model. PyMC is improving this — compiled model functions are increasingly cached at the model level so repeat work on structurally identical graphs can be skipped.

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
