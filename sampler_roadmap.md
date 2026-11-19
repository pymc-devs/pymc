# Roadmap

This document specifies the planned architecture for refactoring `pymc.sample` around a small set of public, typed components:

- steps, with their states and compiled bundles;
- initialization strategies;
- adaptation schedules;
- samplers, which run the steps;
- the output path to storage and monitors.

It is ordered top down: the implementation plan, what the refactor gains (including the naming of its two central abstractions), how `pymc.sample` uses the components, and then each component in turn, from steps down to cross-chain communication and plugins.

**Status:** design draft. All code is pseudocode.

## Implementation order and milestones

The milestones are ordered by dependency, and each one leaves `main` releasable. **`pymc.sample` stays backwards compatible throughout:** existing code keeps working unchanged at every milestone.

| Milestone | Scope | Depends on | Exit criteria |
|---|---|---|---|
| **M0** Contracts | A `pymc.sampling.api` module with the dataclasses and ABCs in this document, and `API_VERSION = 1`. No behaviour change. | — | The types import, and this document matches them. |
| **M1** Canonical loop | `InlineSampler`; the phase and schedule machinery; a `LegacyStep` adapter so every existing `BlockedStep` runs in the new loop; conformance kit v0 (contract, purity, resumability, invariance). | M0 | All existing step methods sample through `InlineSampler` via `LegacyStep`. A ported random-walk Metropolis passes the kit. |
| **M2** HMC split *(optional)* | NUTS and HMC ported as stateless steps. `DualAveraging`, `WindowedDiag`, `WindowedFull`, … extracted from `QuadPotential*`. `InitStrategy`, plus translation of the `init=` strings. | M1 | Posterior regression tests match legacy NUTS. Every legacy `init` string maps to an `(InitStrategy, Schedule)` pair. |
| **M3** Output and monitoring | `DrawBatch` and bundle `postprocess`. The Storage and Monitor contracts, finalized here. Storage adapters for the zarr and ndarray backends. Monitor objects: a progress bar matching today's, a logging monitor, and the adapter that keeps `callback=` working. | M1 | Steps run through `LegacyStep` produce the same `DataTree` on the old and new output paths. Today's progress bar and `callback=` behave as before. |
| **M4** Process sampler and non-blocking API | `ProcessSampler` with free-running workers; `RunHandle` with pause, resume, stop, cancel and checkpoint; resume from storage; `start_sampling` and `sample_async`. `pymc.sample` is routed through the new stack, with its signature, defaults and blocking behaviour unchanged. | M3; M2 if included | The existing test suite passes unchanged on the new stack. A run resumed after `stop()` is bitwise identical to an uninterrupted one (same machine and backend). |
| **M5** Plugins and external samplers | The entry-point registry with its security rules; nutpie, numpyro and blackjax as loop-level `Sampler`s; the new `sampler=` argument, with `nuts_sampler=` kept as an alias; sampler conformance checks. | M3, M4 | `nuts_sampler="nutpie"` and `sampler="nutpie"` produce the same run. The three adapters pass the kit. |
| **M6** Population tier | `Comm` (inline and process), consistent cuts, SMC, ChEES, and DEMetropolis moved off `PopulationStepper`. | M4 | SMC and DEMetropolis results match legacy. Runs with 1, 2 and 3 shards agree. |
| *Not included, but made possible for the future* | A thread-based sampler (free-threaded Python); loops lowered to JAX or numba; mixing chain and population steps in one compound. | — | — |

**If M2 is left out**, NUTS and HMC keep their adaptation inside `QuadPotential*` and run through `LegacyStep`. `LegacyStep` maps the end of the warmup phase to today's `stop_tuning()`. The `init=` strings keep going through today's `init_nuts`. Everything else still works, because M3–M6 don't depend on M2. What waits for M2 is composable adaptation for HMC: custom schedules and adapters, and `InitStrategy` objects passed to `init=`.

## What we gain

### Naming: `Step` and `Sampler` (open for discussion)

The design rests on two abstractions. This section proposes names for them and offers an alternative.

**What they are**

- **`Step`**: one transition of a Markov chain, or of a whole population for methods such as SMC.
  - It consists of configuration plus pure functions; the central one is `transition(state, bundle) -> (state, stats, signals)`.
  - Examples: NUTS, Metropolis, Slice, one SMC stage, a wrapped blackjax kernel.
  - It knows nothing about how many iterations it will run, whether it is tuning, processes, storage or progress bars.
- **`Sampler`**: runs steps to produce samples. It owns the loop:
  - the schedule of phases and adapters, and the stop conditions;
  - parallelism, and one RNG stream per chain;
  - writing to storage and reporting to monitors;
  - the pause, stop and cancel controls.

  Examples: `InlineSampler`, `ProcessSampler`, or the whole loop of an external library (`NutpieSampler`).

**Why they are separate**

A step is a mathematical object, a Markov kernel, so it can be tested in isolation; that is what the conformance kit does. A sampler is an engineering object: concurrency, I/O and control. Keeping them apart has two consequences:

- Any step runs under any sampler that supports it.
- Extension authors have a clear choice: write a `Step` for a new algorithm, or a `Sampler` for a new way of running algorithms (an external library's loop, a cluster, a GPU loop).

**Why these names make sense**

1. **They continue PyMC's own vocabulary.** Users already pass `step=pm.NUTS(...)` to `pm.sample`, and the module is called `pymc.step_methods`. The new `Step` is what `step=` has always accepted, now with a formal contract.
2. **They continue the vocabulary users already apply to external libraries.** `nuts_sampler="nutpie"` selects "the nutpie sampler", and `sampler="nutpie"` selects the same thing.
3. **They read literally.** A sampler is what produces samples, and `pm.sample` hands the work to a `Sampler`. A step is one move of the chain.
4. **Splitting an algorithm into `init` and `step` functions is familiar from blackjax.**

**What they cost**

- **Clashes with legacy names during migration.** `BlockedStep`, `ArrayStep` and `CompoundStep` already exist in `pymc.step_methods`. The new types need their own namespace (`pymc.sampling.api`). The new compound type also needs a name that doesn't shadow the legacy `CompoundStep` while both exist, for example `GibbsStep`. This document calls it `CompoundStep` for readability.
- **The transition method can't also be called `step`**, or generic code reads `step.step(state, bundle)`. This document calls it `transition`.
- **Elsewhere, "sampler" often means the algorithm**, as in "the NUTS sampler", or emcee's `EnsembleSampler`. Some readers will expect `Sampler` to be the algorithm.

**The alternative: `Sampler` for the algorithm, `Executor` for the loop**

- **For:** no clash with legacy step classes; "sampler" matches its common use for the algorithm; "executor" says precisely that it runs work.
- **Against:**
  - `Executor` suggests `concurrent.futures.Executor` and its `submit`/`map` contract, which this isn't.
  - Most PyMC users don't know the word in this sense.
  - It breaks continuity with `step=` and `nuts_sampler=`.

Other libraries use `Kernel` or `TransitionKernel` for the algorithm (numpyro, TensorFlow Probability) and a runner for the loop (numpyro's `MCMC`).

| Concept | Proposed | Alternative |
|---|---|---|
| One transition | `Step` | `Sampler` |
| Its state | `StepState` | `SamplerState` |
| Its transition method | `transition` | `step` |
| A population transition | `PopulationStep` | `PopulationSampler` |
| Gibbs composition of blocks | `CompoundStep` (or `GibbsStep`) | `CompoundSampler` |
| The loop | `Sampler` | `Executor` |
| Its declared capabilities | `SamplerCapabilities` | `ExecutorCapabilities` |
| PyMC's loops | `InlineSampler`, `ProcessSampler` | `InlineExecutor`, `ProcessExecutor` |
| nutpie's loop | `NutpieSampler` | `NutpieExecutor` |
| `pymc.sample` arguments | `step=` (existing), `sampler=` (new) | `step=` (existing, now taking a `Sampler`), `executor=` (new) |
| Entry-point groups | `pymc.steps`, `pymc.samplers` | `pymc.samplers`, `pymc.executors` |

### 1. An extension path that needs no change to PyMC

A developer can write either of two things in their own package:

- a `Step`: one transition, which PyMC's samplers run;
- a `Sampler`: a whole sampling loop, such as an external library's.

They declare an entry point, and `pymc.sample(..., sampler="their-name")` runs it.

As a result:

- PyMC no longer hard-codes `nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"]`.
- Integration code stops accumulating in `pymc/sampling/mcmc.py`.
- A conformance kit tells authors whether their implementation honours the contracts.

This gives us something completely new: **a way for the community to build their own PyMC sampling extensions, and use our models without having to write any translation code**. It's almost as numpy's array interface that lets other libraries define their concrete backends.

### 2. Initialization and adaptation that compose and are validated

Today, `init="jitter+adapt_full"` fuses an initialization method and an adaptation scheme into one string, and adaptation is hard-wired inside `BaseHMC.astep` and `QuadPotentialDiagAdapt.update`.

In the new design, initialization strategies, adapters and phases are separate objects. A schedule such as diagonal -> strongly regularized dense -> full dense is just a list of phases. These combinations are rejected before sampling starts whenever:

- an adapter targets a parameter the step doesn't have.
- an adapter reads a signal the step doesn't emit.
- adaptation is left running during recorded draws without a validity guarantee.

### 3. Validated step arguments

Each step declares:

- the typed state it carries.
- the compiled bundle it needs.
- the parameters it exposes for tuning.
- the space it works in: transformed or not, gradient-based or not, continuous or discrete.

A mismatch is an error at setup, not NaNs or divergences halfway through a run.

### 4. A `pymc.sample` that stops growing, without breaking anyone

Every existing argument keeps working, with the same defaults and the same results. New functionality arrives as components (`step`, `init`, `schedule`, `sampler`, `storage`, `monitor`) instead of more keyword arguments, and the existing arguments are translated into those components internally. Advanced users get the full power of the components. The plan is to eventually remove older arguments through deprecation and deletion cycles.

### 5. Non-blocking sampling, with `pymc.sample` unchanged

`pymc.sample` keeps its current behaviour: it blocks until sampling finishes and returns the same result as today.

The non-blocking alternative is `start_sampling`. It takes the same arguments and immediately returns a `RunHandle`, which offers progress, pause, resume, graceful stop, cancel and checkpoint. `sample_async` awaits the same handle.

One handle type covers both PyMC's samplers and external ones, and each sampler declares which controls it supports.

### 6. Better monitoring, and resumable sampling

Step states are explicit, serializable and self-contained, RNG included, so storage can checkpoint a run and resume it later.

Monitors receive typed events and declared statistics instead of per-step callbacks. Progress display, logging and early-stopping rules (for example, stop at an ESS target) all become pluggable.

### 7. Cross-chain methods as first-class citizens

ChEES-style pooled adaptation, SMC, ensemble samplers and tempering share one population contract with explicit collective communication. This replaces both the $O(\text{chains}^2)$ broadcast in `PopulationStepper` and the separate SMC process machinery.

### Out of scope for now

- Thread-based samplers. v1 parallelism is process-based; see [Threads: not supported in v1](#threads-not-supported-in-v1).
- Compound steps that mix chain-level and population-level steps.
- Loops lowered to JAX or numba. The design leaves room for them; see the last section.

## How the pieces fit

```
                            user code
                                │  pymc.sample (blocking) / start_sampling / sample_async
                                ▼
┌──────────────────────── setup, in the calling process ────────────────────────┐
│ existing arguments ──► components (Step, InitStrategy, Schedule, Sampler, ...) │
│ Model ──model_transform──► Step.Bundle.build ──► CompiledBundle (+ModelSpace)  │
│ InitStrategy ──► starting positions + Approximation (the "hint")               │
│ Step.default_params(bundle, hint) ──► one StepState per chain                  │
│ Schedule = [Phase(adapters, until, warmup), ...]   validated against the Step  │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │ RunSpec (picklable)
                                    ▼
Sampler.start(spec, storage, monitor) ──► RunHandle  (progress, pause, stop, cancel, result)
   runs the canonical loop for each chain, or each group of coupled chains:
     state, stats, signals = step.transition(state, bundle)
     adapters turn signals into new params
     DrawBatch ──► bundle.postprocess ──► Storage.write
     events    ──► Monitor
```

| Component | Role | Written by |
|---|---|---|
| `Step` | Configuration plus a stateless transition: `transition(state, bundle) -> (state, stats, signals)` | PyMC; extension authors |
| `StepState` | Everything that changes between steps: position, tunable parameters, caches, RNG | step author |
| `CompiledBundle` | The compiled callables a step needs, plus a description of the space they work in | step author, usually by reusing a PyMC bundle |
| `InitStrategy` | Starting positions and an optional approximation of the posterior | PyMC, users |
| `Adapter` | Tunes one parameter field from the step's signals | PyMC, users |
| `Phase`, `Schedule` | Which adapters run, for how long, and whether the draws count as warmup | PyMC (defaults), users |
| `StopCondition` | When a phase ends: iterations, ESS, wall-clock time | PyMC, users |
| `Sampler` | Runs the loop: inline, in processes, or inside an external library | PyMC; extension authors |
| `RunHandle` | The user's view of a running sampler | sampler |
| `DrawBatch` | The common output format every sampler emits | sampler |
| `Storage`, `Monitor` | Where draws go; how progress is reported | PyMC, users |

## `pymc.sample`

### Backwards compatibility

- `pymc.sample` keeps its full current signature, its defaults and its return types: a `DataTree`, or a `MultiTrace` with `return_inferencedata=False`. It blocks until sampling finishes, as today.
- **New arguments are added alongside the existing ones:** `sampler`, `schedule`, `storage`, `monitor` and `model_transform`. Two existing arguments also accept the new component types: `step` takes a `Step`, and `init` takes an `InitStrategy`.
- **Adding the new names can't break existing code.** Today an unrecognised keyword ends up in `step_kwargs`, and `assign_step_methods` raises `ValueError` for it (`pymc/sampling/mcmc.py:239-250`). So no working code passes any of the new names.
- **Every call goes through one translation function**, which turns the arguments into components and then follows the same path as `start_sampling`.
- **Disagreement is an error.** Passing an existing argument together with the component that supersedes it fails when they disagree, for example `cores=4` with `sampler=ProcessSampler(cores=2)`.
- **No deprecations are planned as part of this work.** Whether any argument should be deprecated later is listed under open questions.

### Signature

```python
def sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: int | None = None,
    cores: int | None = None,
    random_seed: RandomState = None,
    progressbar: bool | ProgressBarOptions = True,
    progressbar_theme: Theme | None = default_progress_theme,
    quiet: bool = False,
    step: Step | BlockedStep | Sequence[Step | BlockedStep] | None = None,   # now also takes a Step
    var_names: Sequence[str] | None = None,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] | None = None,  # alias for sampler=
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    init: str | InitStrategy = "auto",                                        # now also takes an InitStrategy
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: TraceOrBackend | None = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: bool = True,
    idata_kwargs: dict[str, Any] | None = None,
    nuts_sampler_kwargs: dict[str, Any] | None = None,
    callback=None,
    mp_ctx=None,
    model: Model | None = None,
    blas_cores: int | None | Literal["auto"] = "auto",
    compile_kwargs: dict | None = None,
    # new
    sampler: Sampler | str | None = None,
    schedule: Schedule | None = None,
    storage: Storage | None = None,
    monitor: Monitor | None = None,
    model_transform: Callable[[Model], Model] | None = None,
    **kwargs,                                   # still forwarded to step methods, as today
) -> DataTree | MultiTrace:
    return start_sampling(...same arguments...).result()     # blocks; re-raises errors; Ctrl+C → stop()


def start_sampling(...same arguments...) -> RunHandle:          # the non-blocking alternative
    ...


async def sample_async(...same arguments...) -> DataTree | MultiTrace:
    return await start_sampling(...same arguments...).wait_async()
```

- **`sample` is exactly `start_sampling(...).result()`.** There is one implementation, so the blocking and non-blocking paths can't drift apart.
- **`draws` and `tune` only feed `step.default_schedule(tune, draws)`.** Passing an explicit `schedule` together with non-default `draws` or `tune` is an error; they are never silently ignored.

### Body

```python
def start_sampling(draws=1000, *, model=None, step=None, sampler=None, init="auto", schedule=None,
                   storage=None, monitor=None, model_transform=None, **arguments) -> RunHandle:
    model = modelcontext(model)

    # 0. Translate the existing arguments into components; raise if they disagree with explicit ones
    c = components_from_arguments(model, draws=draws, step=step, sampler=sampler, init=init,
                                  schedule=schedule, storage=storage, monitor=monitor, **arguments)

    # 1. Resume: storage that already holds a run supplies everything except the model
    if c.storage is not None and (checkpoint := c.storage.load_checkpoint()) is not None:
        reject_conflicts(checkpoint, c)
        spec = checkpoint.to_run_spec(model)        # rebuilds the bundle; checks its fingerprint
        sampler = resolve_sampler(c.sampler or checkpoint.sampler_name)
        return sampler.start(spec, c.storage, c.monitor or default_monitor(sampler.capabilities))

    # 2. Components
    sampler = resolve_sampler(c.sampler)                # str → plugin registry (see "Plugins")
    step = c.step
    if step is None:
        step = sampler.default_step(model) if sampler else assign_steps(model)     # competence
    sampler = sampler or step.default_sampler()
    sampler.check_supports(step)

    # 3. Compile once, in this process
    bundle = step.Bundle.build(model, step.vars, transform=model_transform,
                              compile_kwargs=c.compile_kwargs)
    check_space(step.requirements, bundle.space)

    # 4. Initialization
    chains = c.chains or sampler.default_chains()
    seeds = np.random.SeedSequence(c.random_seed).spawn(chains + 1)  # one stream per chain, one for init
    init = c.init or step.default_init()
    init_bundle = bundle if isinstance(bundle, init.Bundle) else init.Bundle.build(model, transform=model_transform)
    init_result = init(init_bundle, chains, rng=np.random.default_rng(seeds[-1]))
    params = step.default_params(bundle, init_result.approximation)
    states = {ch: step.State.new(position=init_result.positions[ch], params=params,
                                rng=np.random.default_rng(seeds[ch]))
              for ch in range(chains)}

    # 5. Schedule
    schedule = c.schedule or step.default_schedule(tune=c.tune, draws=c.draws)
    check_schedule(schedule, step, sampler.capabilities)  # targets, signals, validity

    # 6. Output, then start
    storage = c.storage or default_storage(model, bundle.space)
    monitor = c.monitor or default_monitor(sampler.capabilities)
    spec = RunSpec(step=step, bundle=bundle, states=states, schedule=schedule,
                   hint=init_result.approximation)
    storage.begin(spec)
    return sampler.start(spec, storage, monitor)
```

The flow is the same for PyMC's steps and for external libraries. An external loop-level library provides a *descriptor step*, such as `NutpieNUTS(target_accept=0.9)`. It declares a bundle type and a default schedule, but only its own sampler can run it; see [Plugins: external steps and samplers](#plugins-external-steps-and-samplers). So `pm.sample(step=NutpieNUTS())`, `pm.sample(sampler="nutpie")` and today's `pm.sample(nuts_sampler="nutpie")` all resolve to the same run.

### How existing arguments are translated

| Existing argument | Translated into |
|---|---|
| `draws`, `tune` | the default schedule, `step.default_schedule(tune, draws)` |
| `chains`, `random_seed`, `compile_kwargs` | used as today |
| `step`, `**kwargs` (`target_accept`, `nuts={...}`) | the `Step`; legacy `BlockedStep`s are wrapped in `LegacyStep` |
| `nuts_sampler`, `nuts_sampler_kwargs` | `sampler="nutpie"`, i.e. `NutpieSampler(**nuts_sampler_kwargs)` |
| `init`, `n_init`, `jitter_max_retries`, `initvals` | an `InitStrategy` plus the matching schedule (table below) |
| `cores`, `mp_ctx`, `blas_cores` | `ProcessSampler(cores=..., mp_ctx=..., blas_cores=...)` |
| `progressbar`, `progressbar_theme`, `quiet`, `callback` | the default monitor |
| `trace`, `var_names`, `discard_tuned_samples`, `keep_warning_stat`, `return_inferencedata`, `idata_kwargs` | the default storage |
| `compute_convergence_checks` | a post-run report on the monitor |

With M2, each `init` string becomes an `(InitStrategy, Schedule)` pair, as in the table below. NUTS always also gets `DualAveraging` for the step size. Without M2, the strings keep going through today's `init_nuts`.

| `init=` today | `InitStrategy` | Metric adaptation in the tuning phase |
|---|---|---|
| `"adapt_diag"` | `InitialPoint()` | `WindowedDiag()` (today's foreground/background window scheme) |
| `"jitter+adapt_diag"` (`"auto"`) | `Jitter(max_retries=10)` | `WindowedDiag()` |
| `"jitter+adapt_diag_grad"` | `Jitter()` | `WindowedDiagGrad()` (today's `QuadPotentialDiagAdaptExp`) |
| `"advi+adapt_diag"` | `ADVI(n=200_000)` | `WindowedDiag()`, seeded from the ADVI hint |
| `"advi"`, `"advi_map"` | `ADVI()`, `MAP() >> ADVI()` | none: a fixed diagonal metric from the hint |
| `"map"`, `"jitter+map"` | `MAP()`, `MAP(jitter=True)` | none: a fixed dense metric from the Hessian at the MAP |
| `"adapt_full"`, `"jitter+adapt_full"` | `InitialPoint()`, `Jitter()` | `WindowedFull()` |

### Blocking, non-blocking and resuming

```python
# Unchanged: blocks until done and returns the same result as today
idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)

# The non-blocking alternative: same arguments, returns immediately
handle = pm.start_sampling(1000, chains=4, storage=ZarrStorage("run.zarr"))
handle.progress()        # per chain: iteration, phase, selected stats
handle.pause(); handle.resume()
handle.stop()            # graceful: stops at a consistent point, keeps draws, saves states
idata = handle.result()  # blocks; re-raises worker errors

# From a coroutine, e.g. in a Jupyter cell or a web server
idata = await pm.sample_async(1000, chains=4)

# Later, possibly in a new Python process: continue where the run stopped
idata = pm.sample(storage=ZarrStorage("run.zarr"))
```

## Steps

### Why the current step classes are replaced, not reused

Today's step methods (`BlockedStep`, `ArrayStepShared`, `GradientSharedStep` and their subclasses) already have some of the pieces a `Step` needs:

- a `competence` classmethod;
- declared `stats_dtypes_shapes`;
- a `sampling_state` property that can be read and written.

What they lack is a separation between **configuration**, **state** and **compiled functions**. All three live on one mutable object, which the sampling loop drives through side effects.

**Holding a compiled `logp_dlogp` is not the problem in itself**: holding compiled callables is exactly what a `CompiledBundle` does. The problems are how that function is built and fed (points 1 and 2 below), and the state stored alongside it (points 3 to 7).

1. **Compiled at construction, from a model.**
   - `GradientSharedStep.__init__` calls `modelcontext(model)` and `model.logp_dlogp_function(...)` (`pymc/step_methods/arraystep.py:189-205`).
   - `Metropolis.__init__` builds its `delta_logp` function from `model.logp()` (`pymc/step_methods/metropolis.py:262-263`).

   So a step can't exist without a model, and compile options are fixed when the object is created. Configuration such as `target_accept`, the compiled functions, and references to the model travel together as one object.

2. **Conditioning on other blocks through hidden shared variables.** In a compound step, each block conditions on the others by writing into shared variables inside its compiled function:
   - Before every `astep`, `ArrayStepShared.step` calls `shared_var.set_value(point[name], borrow=True)` for each variable of the other blocks (`arraystep.py:107-111`).
   - For gradient-based steps, those shared variables are the `_extra_vars_shared` of the `ValueGradFunction` (`pymc/model/core.py:234-279`).

   The output of `astep(q0)` therefore depends on values that are in neither its arguments nor `sampling_state`. Two callers sharing the function overwrite each other's values. This is the specific sense in which the link to the compiled function is stateful.

3. **Evolving state stored as attributes.**
   - Metropolis keeps `scaling`, `steps_until_tune` and `accepted_sum` on the instance.
   - HMC keeps the dual-averaging state in `self.step_adapt`, the mass-matrix estimate inside `self.potential`, and counters such as `iter_count` and `_num_divs_sample` (`pymc/step_methods/hmc/base_hmc.py:155-191`).

   `sampling_state` was retrofitted to copy those attributes in and out, and its dataclasses mix every kind of field. `MetropolisState`, for example, holds all of these side by side (`metropolis.py:125-140`):
   - evolving state: `scaling`, `accepted_sum`;
   - configuration: `tune_interval`;
   - the tuning flag: `tune`;
   - derived constants marked `frozen`: `discrete`, `elemwise_update`, `mode`.

4. **One instance reused across chains.** Because the object *is* the state, extra bookkeeping is needed to keep chains apart:
   - Sequential sampling saves `step.sampling_state` and restores it before every chain (`pymc/sampling/mcmc.py:1423-1435`).
   - `setup_chain` documents that implementations "must reset any per-chain state instead of accumulating it" (`compound.py:233-238`).
   - Metropolis-family steps also carry `_untuned_settings` and `reset_tuning()` to undo their own adaptation.

   Missing any of this leaks state from one chain into the next.

5. **Tuning fused into the step.** The step knows whether it is tuning: `self.tune`, `stop_tuning()`, `setup_chain(rng, tune, draws)`. It also adapts itself inside `astep`:
   - Metropolis rescales its proposal every `tune_interval` steps.
   - HMC calls `self.step_adapt.update(...)` and `self.potential.update(...)` (`base_hmc.py:243-244`).

   Changing the adaptation means subclassing the step or the potential.

6. **Randomness held in more than one place.** The step holds `self.rng`, which `setup_chain` sets, and HMC gives its potential a second generator, `rng=self.rng.spawn(1)[0]` (`base_hmc.py:186`).

7. **Cross-chain data through object references.** Population steps call `link_population(population, chain_index)` and keep a reference to a list that `PopulationStepper` overwrites on every iteration (`arraystep.py:154-165`). That only works if every chain can see the same Python list: either all chains run in one process, or the stepper sends every chain's point to every other chain, which is the O(chains²) broadcast it does today.

8. **Constructors that return another type.** When `blocked=False` and several variables are given, `BlockedStep.__new__` returns a `CompoundStep` instead of an instance of the class, and `__getnewargs_ex__` exists so that this still pickles (`compound.py:132-206`). That's convenient, but a contract built on it can't promise that `Metropolis(...)` returns a `Metropolis`.

**Why replace the classes instead of refactoring them in place**

Fixing these problems changes the contract that every subclass relies on:

- `astep(q0)` would need the state as an argument;
- `self.tune` and `stop_tuning()` would disappear;
- `ArrayStepShared(vars, shared)` would stop taking shared variables;
- `link_population` would go away.

Third-party code, such as pymc-extras and user-defined steps, subclasses these classes, so changing them in place breaks every subclass at once. And making a step stateless means rewriting every `self.scaling` as `state.params.scale` anyway.

A parallel hierarchy avoids both problems:

- **Existing steps keep working unchanged** through `LegacyStep` (M1), which meets the backwards-compatibility requirement.
- **Steps are ported one at a time:** random-walk Metropolis in M1, then NUTS and HMC in the optional M2.
- **The numerical cores are reused, not rewritten:** proposal distributions, `metrop_select`, the leapfrog integrator and NUTS tree building, the slice logic, `competence`, and the `stats_dtypes_shapes` declarations, which become `stats`. The one structural change inside them is that the integrator receives the metric as a parameter instead of holding a stateful potential.
- **The old classes are retired on their own schedule**, once their replacements exist.

**What `LegacyStep` can and can't do**

`LegacyStep` lets the new loop run an unmodified legacy step:

- **The legacy object itself serves as the state.** It is pickled whole at checkpoints and copied through `sampling_state`.
- **The end of warmup maps to `stop_tuning()`.**
- **The step's own adaptation runs exactly as before.**

What it can't do:

- It can't take adapters from a schedule.
- It relies on the sampler giving each chain its own copy of the object. `ProcessSampler` does that, and `InlineSampler` does it by restoring `sampling_state`, as `_sample_many` does today.

### The contract

```python
class Step(ABC, Generic[StateT, BundleT, ParamsT, SignalsT]):
    State: ClassVar[type[StateT]]
    Bundle: ClassVar[type[BundleT]]
    Params: ClassVar[type[ParamsT]]          # what adapters may tune
    Signals: ClassVar[type[SignalsT]]        # what each step offers to adapters
    stats: ClassVar[Mapping[str, Stat]]      # every statistic transition() returns
    requirements: ClassVar[Requirements]

    def __init__(self, vars=None, **config): ...         # immutable configuration only

    # Setup
    @classmethod
    def competence(cls, var, has_grad) -> Competence: ...
    def default_params(self, bundle: BundleT, hint: Approximation | None) -> ParamsT: ...
    def default_init(self) -> InitStrategy: ...
    def default_schedule(self, tune: int, draws: int) -> Schedule: ...
    def default_sampler(self) -> Sampler: return ProcessSampler()

    # Sampling: pure functions of (state, bundle)
    def init(self, state: StateT, bundle: BundleT) -> StateT: ...
    def transition(self, state: StateT, bundle: BundleT) -> tuple[StateT, Stats, SignalsT]: ...
    def is_finished(self, state: StateT) -> bool: return False
    def finalize(self, state: StateT, bundle: BundleT) -> Records: return {}


@dataclass(frozen=True)
class Stat:
    dtype: np.dtype
    shape: tuple[int | None, ...] = ()
    arviz_name: str | None = None      # e.g. "diverging", "tree_depth", "energy"
    monitor: bool = False              # shown in progress reports by default
    warning: bool = False              # e.g. divergence details, routed to the monitor's warnings


@dataclass(frozen=True)
class Requirements:
    space: Literal["unconstrained", "any"] = "any"
    gradient: bool = False
    values: Literal["continuous", "discrete", "any"] = "any"
```

**Rules**

1. The step object holds configuration only (`target_accept`, `max_treedepth`, …). It never caches anything computed during sampling. It must be picklable and cheap to copy. That makes it a coordinator, not a container of state.
2. `transition` is a pure function of `(state, bundle)`; the one exception is advancing the state's RNG (see [RNG and ownership](#rng-and-ownership)).
3. **A step doesn't know whether it is tuning.** It only sees `state.params`. Whether those params are still being adapted is the adapters' business, and whether draws count as warmup is the phase's.
4. `init` fills the caches derived from the position (log density, gradient). It is idempotent, so it serves fresh and resumed states alike.
5. The step never touches storage, monitors, processes or other chains. Statistics leave as return values, and the sampler routes them.
6. `stats` declares every statistic with its dtype and shape, so storage can preallocate and monitors know what is available.

### Example: random-walk Metropolis

This is everything an author writes for a simple step.

```python
@dataclass(frozen=True)
class RWParams:
    scale: np.ndarray                         # (ndim,)

@dataclass(frozen=True)
class RWSignals:
    accept_prob: float

@dataclass(frozen=True)
class RWState(StepState[RWParams]):
    logp: float | None = None                 # cache, filled by init()


class RandomWalk(Step[RWState, LogpBundle, RWParams, RWSignals]):
    State, Bundle, Params, Signals = RWState, LogpBundle, RWParams, RWSignals
    stats = {"accepted": Stat(np.bool_),
             "accept_prob": Stat(np.float64, arviz_name="acceptance_rate", monitor=True)}
    requirements = Requirements(space="any", gradient=False, values="continuous")

    def default_params(self, bundle, hint):
        return RWParams(scale=hint.cov.std() if hint else np.ones(bundle.space.ndim))

    def default_init(self):
        return Jitter()

    def default_schedule(self, tune, draws):
        return [Phase("tune", (AcceptanceScale(target_rate=0.234),), Iterations(tune)),
                Phase("sample", (), Iterations(draws), warmup=False)]

    def init(self, state, bundle):
        return state.replace(logp=bundle.logp(state.position, state.cond))

    def transition(self, state, bundle):
        noise = state.rng.standard_normal(state.position.shape)
        proposal = state.position + state.params.scale * noise
        logp_new = bundle.logp(proposal, state.cond)
        log_alpha = min(0.0, logp_new - state.logp)
        accepted = np.log(state.rng.uniform()) < log_alpha
        new = state.replace(position=proposal, logp=logp_new) if accepted else state
        new = new.replace(iteration=state.iteration + 1)
        accept_prob = float(np.exp(log_alpha))
        return new, {"accepted": accepted, "accept_prob": accept_prob}, RWSignals(accept_prob)
```

The same step can be tuned differently without touching its code. For example:

```python
schedule = [
    Phase("tune", (AcceptanceScale(),), Iterations(500)),
    Phase("sample", (), Iterations(2000), warmup=False),
]
```

### Termination

Termination comes in two kinds, and they live in different places:

- **Intrinsic** termination is part of the algorithm. Examples: SMC reaching β = 1; nested sampling reaching its evidence tolerance. The step implements it as `is_finished(state)`.
- **Extrinsic** termination is a budget or a goal: a number of draws, an ESS target, wall-clock time. It is the final phase's `until=` condition (see [Stop conditions](#stop-conditions)).

The loop checks both after every step. If an algorithm has outputs that only exist at the end, such as SMC's final population and its marginal-likelihood estimate, it produces them in `finalize(state, bundle)`.

### Compound steps

- `CompoundStep((NUTS(vars=[mu, sigma]), Slice(vars=[k])))` is itself a `Step`.
  - Its state is a tuple of sub-states.
  - Its bundle is a tuple of sub-bundles, all built from the same transformed model.
  - Its `transition` is a Gibbs sweep: before each block steps, that block receives the other blocks' current values in `state.cond`.
- Changing `cond` invalidates the block's cached log density, so the compound runs that block's `init` before its `transition`. This is no more expensive than today: legacy step methods recompute the log density at the start of every step.
- External steps, such as a blackjax kernel wrapped as a `Step`, compose the same way. When blocks use different array libraries, `cond` is converted at block boundaries (zero-copy via DLPack where possible).
- Adapters bind to a block: `DualAveraging().on(nuts_block)`.
- Out of scope for v1: mixing chain steps and population steps in one compound.

### Population steps

```python
class PopulationStep(Step[PopStateT, BundleT, ParamsT, SignalsT]):
    shardable: ClassVar[bool] = False        # may members be split across workers?

    def transition(self, state: PopStateT, bundle: BundleT, comm: Comm
                   ) -> tuple[PopStateT, Stats, SignalsT]: ...


@dataclass(frozen=True)
class PopulationState(Generic[ReplicatedT, MembersT]):
    replicated: ReplicatedT    # identical on every shard: SMC's beta and stage, a pooled metric, a shared RNG
    members: MembersT          # leading axis = the members this shard holds: positions, log densities, weights
```

The only difference from a chain step is the `comm` argument. It gives access to collective operations over the whole population (see [Cross-chain communication](#cross-chain-communication)). A population step running unsharded on one worker receives an identity `comm`.

### Continuous adaptation: GHMC, MEADS and adaptive Metropolis

The previous draft required the final phase to have no adapters. That rule is too strict. Three different things get called "continuous adaptation", and the framework puts each one in a different place.

| Kind | Example | Where it lives | Why the draws are valid |
|---|---|---|---|
| **Part of the Markov state** | GHMC's partially refreshed momentum, $p \leftarrow \alpha p + \sqrt{1-\alpha^2}\,\xi$ (Horowitz, 1991) | `GHMCState.momentum`; the damping α in `Params` | The kernel is fixed on the extended space $(x, p)$ and leaves $\pi(x)\,\mathcal{N}(p)$ invariant |
| **Ensemble kernel** | MEADS (Hoffman & Sountsov, 2022): each fold's step size, damping and preconditioner come from the *other* folds' current states. Also stretch moves. | A `PopulationStep`; the "adapted" quantities are computed inside `transition` | It's a fixed kernel on the product space. Each fold's update conditions only on the other folds, so it leaves $\prod_i \pi(x_i)$ invariant. Updating every fold at once from each other's states would not. |
| **History-dependent adaptation** | Adaptive Metropolis (Haario et al., 2001); metric estimators whose updates shrink like $1/n$ | An `Adapter` declaring `validity="diminishing"` | Not invariance: ergodicity holds under diminishing adaptation and containment (Roberts & Rosenthal, 2007) |

So yes: GHMC with persistent momentum is a pure step with an ordinary, non-adaptive sampling phase. That isn't a workaround; it is the correct mathematical description. It also means the conformance kit's invariance check applies to GHMC directly: start from exact draws of $\pi(x)\,\mathcal{N}(p)$.

The old rule becomes this one: **every adapter active in a phase with `warmup=False` must declare a `validity` other than `"warmup_only"`.** The schedule validator rejects any other adapter unless the phase opts out explicitly, and the opt-out is recorded in the output's attributes.

## Step state

```python
@dataclass(frozen=True)
class StepState(Generic[ParamsT]):
    position: np.ndarray                    # this step's block, raveled, in bundle.space
    params: ParamsT                         # tunable parameters; adapters write here
    rng: np.random.Generator                # one independent stream per chain
    iteration: int = 0
    cond: RaveledVars | None = None         # other blocks' values, inside a compound
    bundle_fingerprint: str | None = None   # which bundle the caches were computed with

    @classmethod
    def new(cls, position, params, rng) -> Self: ...
    def replace(self, **changes) -> Self: ...
    def snapshot(self) -> Self: ...         # independent copy, including the RNG's bit-generator state


@dataclass(frozen=True)
class HMCParams:
    step_size: float
    metric: Metric                          # DiagMetric | DenseMetric | LowRankMetric


@dataclass(frozen=True)
class NUTSState(StepState[HMCParams]):
    logp: float | None = None               # caches: None until init()
    grad: np.ndarray | None = None
```

Authors subclass `StepState` and add fields. Anything that changes from one step to the next belongs here and nowhere else.

### RNG and ownership

**What we do today, and keep.**

- `pymc.sample` spawns one `Generator` per chain from the user's seed: `get_random_generator(random_seed).spawn(chains)` in `pymc/sampling/mcmc.py:921`.
- Each worker rebuilds its generator from a `RandomGeneratorState`, because pickling a `Generator` loses its `SeedSequence` (numpy#27727; `pymc/sampling/parallel.py:157-160`).
- The worker then hands the generator to the step method in `setup_chain`.

Independent streams per chain plus one process per chain already give correct parallel sampling, and v1 keeps exactly that. The only change is where the generator lives: it moves from the step method into the state.

**The remaining issue is purity, not threads.** `transition` advances the generator in place, so the state it was given is modified. That is harmless in a worker process, as long as nothing else holds a reference to the old state. Instead of requiring counter-based keys now, the contract states a **linear ownership** rule:

- `transition` and `init` consume the state they are given. The caller must not use that object again.
- Anything that needs to keep a state calls `state.snapshot()` first: checkpoints, consistent cuts, and the conformance kit's purity check.
- In the normal loop this costs nothing.

**Leaving room for later.** Counter-based keys (Philox key + counter, as in JAX) can come later, and are only needed if the loop is lowered to JAX. To make that change non-breaking, everything except the step's own `transition` treats `state.rng` as opaque.

**Hidden RNGs.** Compiled graphs can contain shared random-generator variables with default updates, such as the jitter in the initial-point function. `init` reseeds them from the state's generator, as `reseed_rngs` does today. No randomness may come from anywhere else.

### Serialization, checkpoints and resume

- Every state must round-trip through `pickle`. Storage persists states at checkpoints; the `_sampling_state` group of `ZarrTrace` already exists for this.
- A checkpoint holds three things: the step states, the position in the schedule (phase index and progress within the phase), and the adapter states. Without the adapter states, a run stopped during warmup couldn't resume.
- Resuming means loading a checkpoint, building a fresh bundle, and calling `init` on each state.
- The bundle fingerprint stored in the state detects a changed model or changed data. In that case resume fails, unless the user explicitly accepts the change.
- For population steps, the checkpoint holds the whole `PopulationState`, captured at a consistent cut.

## Compiled bundles

### What a bundle is

```python
class CompiledBundle(ABC):
    space: ModelSpace                     # what the callables operate on
    fingerprint: str                      # hash of the graph, the model transform and the data

    @property
    def functions(self) -> Mapping[str, Callable]: ...   # every compiled callable, by field name

    @classmethod
    @abstractmethod
    def build(cls, model: Model, vars: Sequence[Variable] | None = None, *,
              transform: Callable[[Model], Model] | None = None,
              compile_kwargs: dict | None = None) -> Self: ...

    def postprocess(self, positions: np.ndarray) -> dict[str, np.ndarray]: ...  # unravel, untransform, deterministics
    def clone(self) -> Self: ...          # independent function storage; default: Function.copy() on every field
```

Concrete bundles are typed dataclasses. The fields document exactly which callables exist and with what signature, and `.functions` exposes them as a mapping for generic code.

```python
@dataclass(frozen=True)
class LogpBundle(CompiledBundle):
    logp: Callable[[np.ndarray, RaveledVars | None], float]

@dataclass(frozen=True)
class GradBundle(CompiledBundle):
    logp_dlogp: Callable[[np.ndarray, RaveledVars | None], tuple[float, np.ndarray]]

@dataclass(frozen=True)
class TemperedBundle(CompiledBundle):          # SMC
    prior_logp_batch: Callable[[np.ndarray], np.ndarray]
    loglik_batch: Callable[[np.ndarray], np.ndarray]
```

Most step authors reuse PyMC's bundles. A new bundle type is only needed when a step requires a callable that none of the existing ones provide.

### The space a bundle works in

The bundle records which space its callables operate in, so steps never have to inspect the model to find out whether it was transformed.

```python
@dataclass(frozen=True)
class VarSpace:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    transform: Transform | None     # None → sampled on the original scale
    constrained: bool               # bounded support with no transform applied
    discrete: bool

@dataclass(frozen=True)
class ModelSpace:
    vars: tuple[VarSpace, ...]      # the step's block, in ravel order
    ravel: RavelInfo                # slices and shapes into the flat position vector
    jacobian: bool                  # whether logp includes the transforms' log-Jacobian
    ndim: int
```

What this enables:

- `check_space(step.requirements, bundle.space)` runs at setup. For example, `model_transform=remove_value_transforms` combined with NUTS fails immediately with "`sigma` has bounded support and no transform; NUTS requires an unconstrained space", instead of producing divergences halfway through the run.
- `postprocess` and `DrawBatch.space` use it to map draws back to the original scale, so the output looks the same whichever step or sampler produced it.
- A compound step checks that all its blocks were built from the same transformed model.

### Built once, in the calling process

- `build` runs in the calling process. It applies `model_transform` first, so model mutation never happens inside steps or workers. Examples: `remove_value_transforms`, `do`, `observe`, freezing data, reparameterizations.
- The backend (C, numba, JAX) is chosen through `compile_kwargs` at this point. Steps don't take a `backend` argument.
- Each bundle is pickled once per worker. Unpickling rebuilds the compiled functions, at a cost that depends on the backend:
  - C loads from its compile cache, which is cheap.
  - numba may re-JIT unless its cache is warm.
  - JAX re-traces on the first call.
  - With the `fork` start method, workers inherit the compiled functions and pay nothing.
- PyMC already works this way today: step methods that hold compiled functions are pickled to each worker (`_Process._unpickle_step_method`).

### Rules for the callables' arguments

1. Every per-call input is an explicit argument: the position, other blocks' values (`cond`), β, a batch of particles. This replaces how `ArrayStepShared` conditions on other blocks today, through `shared[var].set_value(...)`, which is hidden mutable state.
2. Model data is fixed for the lifetime of a bundle. Changing data means building a new bundle; the fingerprint changes, and `init` recomputes the caches.
3. Randomness enters only through the state's RNG.
4. Callables never cross a process boundary on every step. The bundle travels once; states travel at the start and at checkpoints; each step sends only small records.

### Threads: not supported in v1

PyTensor `Function`s aren't reentrant. Each one owns its input and output storage and the containers of the shared variables it updates, RNGs included. Two threads calling the same function overwrite each other's values.

Processes avoid the problem, because each worker unpickles its own copy. So the v1 position is: **parallelism is process-based, and thread-based samplers aren't supported yet.**

The contract still includes `clone()`, with a default implementation that calls `Function.copy()` on every field, and it is samplers that decide when to clone, not step authors. A thread-based sampler, useful with free-threaded Python, can then be added later without changing any step or bundle.

## Initialization

```python
@dataclass(frozen=True)
class Approximation:                       # Gaussian, in bundle.space
    mean: np.ndarray
    cov: DiagCov | DenseCov | LowRankPlusDiag


@dataclass(frozen=True)
class InitResult:
    positions: np.ndarray                  # (chains, ndim), in bundle.space
    approximation: Approximation | None = None


class InitStrategy(ABC):
    Bundle: ClassVar[type[CompiledBundle]]

    def __call__(self, bundle, chains: int, rng: np.random.Generator,
                 previous: InitResult | None = None) -> InitResult: ...

    def __rshift__(self, other: "InitStrategy") -> "InitStrategy": ...    # pipeline
```

- **Strategies:** `InitialPoint()`, `InitialValues(initvals)`, `Jitter(scale, max_retries)`, `MAP()`, `ADVI(n)`, `Pathfinder(num_paths)`, or a short run of another step.
- **Pipelines:** in `Pathfinder(num_paths=4) >> Jitter(0.1)`, each stage receives the previous stage's result.
- **Hand-off to adaptation:** the approximation is how initialization passes information to adaptation. `step.default_params(bundle, hint)` and `adapter.init(current, hint)` both read it. Today's `"advi+adapt_diag"` becomes `init=ADVI()` plus a metric adapter seeded from the hint.

## Adaptation and schedules

### Parameters and signals

A step is coupled to adaptation through exactly two declarations: what it lets adapters tune (`Params`) and what each step reports to them (`Signals`).

```python
@dataclass(frozen=True)
class HMCParams:                 # tunable
    step_size: float
    metric: Metric

@dataclass(frozen=True)
class HMCSignals:                # offered to adapters after each step; not stored by default
    accept_stat: float
    position: np.ndarray
    grad: np.ndarray
```

### Adapters

```python
class Adapter(ABC, Generic[ParamT, SignalsT]):
    target: ClassVar[str]                      # the Params field this adapter owns
    requires: ClassVar[frozenset[str]]         # the Signals fields it reads
    validity: ClassVar[Literal["warmup_only", "diminishing"]] = "warmup_only"

    def init(self, current: ParamT, hint: Approximation | None) -> AdaptState: ...
    def update(self, a: AdaptState, signals: SignalsT, comm: Comm) -> AdaptState: ...
    def propose(self, a: AdaptState) -> ParamT: ...     # value used for the next step
    def final(self, a: AdaptState) -> ParamT: ...       # value handed to the next phase
```

- Adapters are pure, like steps. `AdaptState` is serializable and saved in checkpoints.
- `comm` lets an adapter pool its estimate across chains, for example one metric shared by all chains. For independent adaptation it is the identity.
- Because adapters depend only on declared fields, they work with any step that declares those fields. For example, `WelfordDiag` works for HMC, NUTS, MALA, or anything else with a `metric` parameter.

| Adapter | `target` | `requires` |
|---|---|---|
| `DualAveraging(target_accept)` | `step_size` | `accept_stat` |
| `WindowedDiag()`, `WelfordDiag()` | `metric` | `position` |
| `WindowedDiagGrad()`, `FisherDiag()` (nutpie-style) | `metric` | `position`, `grad` |
| `WindowedFull()`, `WelfordDense()` | `metric` | `position` |
| `ShrunkDense(toward="current", shrinkage)` | `metric` | `position` |
| `LowRank(k)` | `metric` | `position`, `grad` |
| `AcceptanceScale(target_rate)` | `scale` | `accept_prob` |

### Phases and schedules

```python
@dataclass(frozen=True)
class Phase:
    name: str
    adapters: tuple[Adapter, ...] = ()          # empty → params frozen
    until: StopCondition = Iterations(0)
    warmup: bool = True                         # draws flagged as warmup in storage
    record: bool = True                         # store this phase's draws at all
    allow_invalid_adaptation: bool = False      # explicit opt-out; recorded in the output

Schedule = Sequence[Phase]                      # or an iterable, for schedules built lazily
```

**Phase boundary rule:** when a new phase starts, each of its adapters is initialized with `init(current=<the previous phase's final value>, hint=...)`. That single rule covers Stan's step-size restart at the end of every window and the hand-off between metric estimators.

```python
# Stan's windowed adaptation
def stan_windows(tune=1000, init_buffer=75, term_buffer=50, base_window=25, metric=WelfordDiag):
    phases = [Phase("fast-init", (DualAveraging(),), Iterations(init_buffer))]
    for i, w in enumerate(doubling_windows(tune - init_buffer - term_buffer, base_window)):
        phases.append(Phase(f"slow-{i}", (DualAveraging(), metric()), Iterations(w)))
    phases.append(Phase("fast-term", (DualAveraging(),), Iterations(term_buffer)))
    return phases

# Diagonal → strongly regularized dense → full dense
schedule = [
    Phase("diag", (DualAveraging(), WelfordDiag()), Iterations(200)),
    Phase("reg",  (DualAveraging(), ShrunkDense(toward="current", shrinkage=0.9)), Iterations(300)),
    Phase("full", (DualAveraging(), WelfordDense()), Iterations(500)),
    Phase("sample", (), Iterations(1000) | EssAtLeast(400), warmup=False),
]
```

`ShrunkDense(toward="current")` shrinks toward whatever the previous phase handed over; here that is the diagonal estimate.

### Validation

`check_schedule(schedule, step, capabilities)` runs at setup and checks that:

1. every `adapter.target` is a field of `step.Params`, with a matching type;
2. every `adapter.requires` is a subset of the fields of `step.Signals`;
3. each phase has at most one adapter per target;
4. every adapter in a phase with `warmup=False` declares `validity != "warmup_only"`, unless the phase sets `allow_invalid_adaptation=True`;
5. run-scoped stop conditions are only used with samplers that can evaluate them;
6. the sampler accepts a custom schedule at all (`capabilities.custom_schedule`). For example, a loop-level external library that runs its own warmup cannot take a user-defined schedule.

### Stop conditions

```python
class StopCondition(Protocol):
    scope: Literal["chain", "run"]
    def done(self, progress: PhaseProgress, run: RunView | None) -> bool: ...
    def __or__(self, other: "StopCondition") -> "StopCondition": ...
    def __and__(self, other: "StopCondition") -> "StopCondition": ...
```

- **Built-in conditions:** `Iterations(n)`, `EssAtLeast(400, var_names=...)`, `RhatBelow(1.01)`, `WallTime(minutes=30)`.
- **Chain-scoped conditions**, such as iterations or wall time per chain, are checked by the loop after every step.
- **Run-scoped conditions**, such as ESS or R-hat, need data from all chains. The sampler evaluates them on the draws that have been stored. When one fires, the sampler stops all chains at a consistent point.

## Samplers

### The canonical loop

This defines the reference semantics. PyMC's samplers implement this loop. An external loop-level sampler must produce the same observable behaviour: the same events, in the same order, and output in the same format.

```python
def run_chain(step, bundle, state, schedule, comm, sink, hint, checkpoint=None):
    state = step.init(state, bundle)
    for phase_idx, phase in enumerate(schedule):
        if checkpoint and phase_idx < checkpoint.phase_idx:
            continue                                        # resume: skip finished phases
        adapters = [(a, a.init(getattr(state.params, a.target), hint)) for a in phase.adapters]
        progress = PhaseProgress.start(phase, checkpoint)
        while not (phase.until.done(progress, sink.run_view) or step.is_finished(state)):
            state, stats, signals = step.transition(state, bundle)
            adapters = [(a, a.update(s, signals, comm)) for a, s in adapters]
            state = state.replace(params=state.params.replace(
                **{a.target: a.propose(s) for a, s in adapters}))
            progress = progress.advance()
            sink.record(phase, state, stats)      # → DrawBatch chunks → storage; progress → monitor
            yield state, adapters, progress       # control point: pause, stop, checkpoint
        state = state.replace(params=state.params.replace(
            **{a.target: a.final(s) for a, s in adapters}))
        sink.phase_end(phase, state)
    sink.final(step.finalize(state, bundle))
```

The `yield` is the only place where a run can be paused, stopped or checkpointed. Everything between two control points is one synchronous unit of work.

### The sampler contract

```python
class Sampler(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[SamplerCapabilities]

    def default_step(self, model: Model) -> Step: ...
    def default_chains(self) -> int: ...
    def check_supports(self, step: Step) -> None: ...      # raises with a clear reason
    def start(self, spec: RunSpec, storage: Storage, monitor: Monitor) -> RunHandle: ...


@dataclass(frozen=True)
class RunSpec:                                 # picklable: everything a run needs
    step: Step
    bundle: CompiledBundle
    states: Mapping[int, StepState]
    schedule: Schedule
    hint: Approximation | None = None
    checkpoint: Checkpoint | None = None       # set when resuming


@dataclass(frozen=True)
class SamplerCapabilities:
    controls: frozenset[Literal["pause", "resume", "stop", "cancel", "checkpoint"]]
    progress: Literal["per_draw", "per_chunk", "end_only"]
    streaming_output: bool        # draws arrive during the run, or only at the end
    custom_schedule: bool         # honours user-defined phases and adapters
    live_stats: bool              # step stats reach the monitor during the run
    runs_pymc_steps: bool         # can run arbitrary pymc Step objects
```

The capabilities are the flags that tell the monitor, the storage and the user what this sampler allows.

### PyMC's samplers in v1

| Sampler | Used for | Notes |
|---|---|---|
| `InlineSampler` | one core, debugging, Pyodide | Runs chains in the calling thread, round-robin. It can be driven synchronously, from an asyncio task (yielding to the event loop between chunks), or one step at a time. |
| `ProcessSampler` | the default when `cores > 1` | One worker process per running chain, up to `cores` at a time; further chains wait their turn. |

How `ProcessSampler` works:

- **Workers run freely**, with no handshake for each draw. Each worker:
  - checks its control inbox at control points without blocking;
  - writes output in chunks, either to shared memory or directly to zarr storage.

  The current `write_next` / `writing_done` handshake only exists because the shared buffer is reused.
- **A controller thread** in the calling process:
  - waits on all worker pipes plus a wake-up pipe;
  - forwards events to the monitor;
  - implements the `RunHandle` controls.

  No asyncio event loop is needed internally, so execution behaves identically in scripts, IPython and Jupyter, which already runs its own loop.
- **Orphan detection:** a worker exits when its pipe to the parent reaches EOF. This covers a parent that was killed; daemon processes only clean up when the parent exits normally.
- **Ctrl+C** during a blocking `pymc.sample` calls `stop()`. A second Ctrl+C calls `cancel()`.

### The run handle

```python
class RunHandle:
    status: Literal["pending", "running", "paused", "stopping",
                    "finished", "cancelled", "failed"]
    capabilities: SamplerCapabilities

    def progress(self) -> Progress: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...          # graceful: consistent point, draws kept, states saved → resumable
    def cancel(self) -> None: ...        # immediate: output so far stays valid; states from the last checkpoint
    def checkpoint(self) -> None: ...
    def add_done_callback(self, fn: Callable[["RunHandle"], None]) -> None: ...
    def result(self, timeout: float | None = None) -> DataTree: ...   # blocks; re-raises worker errors
    async def wait_async(self) -> DataTree: ...                         # asyncio.wrap_future under the hood
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc) -> None: ...                               # cancels if still running
```

- The handle is synchronous and thread-safe. `wait_async` wraps a `concurrent.futures.Future`, so it can be awaited from any running event loop, Jupyter's included.
- Calling a control that is not in `capabilities.controls` raises `UnsupportedControl`.
- **Pyodide** has neither threads nor processes. `InlineSampler` runs as an asyncio task on the browser's event loop and yields between chunks. There, `sample_async` is the entry point, because a blocking `sample` can't yield to the browser.

## Cross-chain communication

### How `comm` works

Population steps follow the SPMD model: every worker runs the **same** `transition` on its own slice of the members. A *collective* is a call where every worker contributes a value, waits until all have arrived, and gets the combined result back. The workers then carry on in parallel.

For example, 6 chains on 3 workers:

```
worker 0 (chains 0–1) ── local work ──► allreduce(g₀) ──┐                ┌──► local work ─► …
worker 1 (chains 2–3) ── local work ───────► allreduce(g₁) ─┤  wait for all; ├──► local work ─► …
worker 2 (chains 4–5) ── local work ──► allreduce(g₂) ──┘  g = g₀+g₁+g₂  └──► local work ─► …
```

So yes: `transition` hides whether the work is distributed. The same step code runs unsharded on one worker and sharded across several.

```python
class Comm(Protocol):
    rank: int                              # this shard
    size: int                              # number of shards
    def allreduce(self, x: np.ndarray, op: Literal["sum", "max", "min"] = "sum") -> np.ndarray: ...
    def allgather(self, x: np.ndarray) -> np.ndarray: ...    # concatenated along the member axis
    def exchange(self, x: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray: ...  # e.g. tempering swaps
```

| Sampler | `Comm` implementation |
|---|---|
| Inline, or any unsharded population | identity: `allreduce(x) == x` |
| Processes | the controller's pipes for small payloads, shared memory for large ones |
| Threads (later) | a shared array plus `threading.Barrier` |
| A loop lowered to JAX (later) | `lax.psum` over a named axis |

**Rules**

1. Every shard calls the same collectives in the same order.
2. `comm` is passed as an argument and never stored.
3. Results agree across shard counts up to floating-point rounding. Bitwise agreement would need allgather-then-reduce in a fixed member order; that is an open question.

### Example: SMC as a population step

```python
class SMC(PopulationStep[SMCState, TemperedBundle, SMCParams, SMCSignals]):
    shardable = True

    def transition(self, state, bundle, comm):
        r, m = state.replicated, state.members
        # 1. Next temperature: bisection on the global ESS (one allreduce per evaluation)
        beta = bisect(lambda b: global_ess(comm, (b - r.beta) * m.loglik) - self.threshold * r.n_total,
                      lo=r.beta, hi=1.0)
        logw = (beta - r.beta) * m.loglik
        # 2. Resample: every shard sees all weights and uses the replicated RNG,
        #    so all shards pick the same indices
        all_logw = comm.allgather(logw)
        idx = systematic_resampling(all_logw, rng=r.rng)
        m = take_members(comm.allgather(m), idx[my_slice(comm, r.n_total)])
        # 3. Tune the proposal from the population covariance, then mutate locally
        cov = global_cov(comm, m.position)
        m = mutate(m, cov, beta, bundle)          # per-member RNGs live in m
        new = state.replace(replicated=r.replace(beta=beta, stage=r.stage + 1), members=m)
        return new, {"beta": beta}, SMCSignals(...)

    def is_finished(self, state):
        return state.replicated.beta >= 1.0
```

### Groups, scheduling and failure

- A **group** is the set of members of one population run: one SMC run, or one set of chains that share a pooled metric.
- **Gang scheduling:** all shards of a group start together, so a collective never waits for a worker that hasn't started yet.
- **Failure domain:** if one shard fails, its whole group fails. Other groups keep running.
- **When to shard:** only when there are fewer groups than workers. Otherwise a batched step on a single worker is cheaper than collectives across processes.

### Consistent cuts

Stopping or checkpointing coupled members requires every shard to stop at the same iteration. The controller uses two phases:

1. **Prepare:** shards in a group meet at every collective, so they are at most one step apart. The controller announces a target iteration `t`.
2. **Apply:** every shard runs until iteration `t`, then saves or stops and acknowledges.

Independent chains need no coordination: each chain stops or checkpoints at its own iteration.

## Output: `DrawBatch`, storage and monitor

### The common output format

Every source of draws, internal or external, emits the same chunk type. Storage only ever sees this type.

```python
@dataclass
class DrawBatch:
    chains: np.ndarray                  # chain ids along the leading axis
    draw_start: int                     # offset along the draw axis
    phase: str
    warmup: bool
    space: Literal["unconstrained_raveled", "unconstrained", "constrained"]
    values: Mapping[str, np.ndarray]    # (chain, draw, *shape)
    stats: Mapping[str, np.ndarray]     # (chain, draw, *shape); ArviZ sample_stats names where they exist
```

The pipeline:

```
external output ──translator──► DrawBatch ──┐
internal samplers ────────────► DrawBatch ──┴──bundle.postprocess──► DrawBatch (constrained, with deterministics) ──► storage.write
```

- External libraries return whatever they like: Arrow tables, dicts of JAX arrays. Their adapter's translator converts that to a `DrawBatch`.
- PyMC owns postprocessing (unraveling, untransforming, deterministics), so coords, dims and deterministics come out the same regardless of which sampler produced the draws.
- Postprocessing runs wherever the batch is written, in a worker or in the calling process.
- Internal samplers emit one batch every N draws. A loop-level library that only reports at the end emits one large batch per chain.

### Storage and monitor (sketch, parked until M3)

```python
class Storage(Protocol):
    def begin(self, spec: RunSpec) -> None: ...
    def write(self, batch: DrawBatch) -> None: ...                   # already postprocessed
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def load_checkpoint(self) -> Checkpoint | None: ...
    def finalize(self) -> DataTree: ...


class Monitor(Protocol):
    def on_start(self, spec: RunSpec, capabilities: SamplerCapabilities) -> None: ...
    def on_progress(self, event: ProgressEvent) -> None: ...         # chain, iteration, phase, monitored stats
    def on_phase_end(self, chain: int, phase: Phase) -> None: ...
    def on_warning(self, warning: SamplingWarning) -> None: ...
    def on_finish(self, status: str, report: RunReport) -> None: ...
```

The loop-level external samplers will be the stress test for both contracts. They report at a coarser granularity, they can't honour every control, and they bring their own output formats.

## Plugins: external steps and samplers

### Two kinds of external integration

| | Implements | Runs in | Examples |
|---|---|---|---|
| **Step level** | `Step`, plus its `State` and `Bundle` | any PyMC sampler; composes with PyMC's steps in a compound | blackjax kernels; a nuts-rs step wrapper |
| **Loop level** | `Sampler`, plus a descriptor step | the library's own runtime | `nutpie.sample`, numpyro's `MCMC` |

A **descriptor step** such as `NutpieNUTS(target_accept=0.9)` declares a bundle type (which wraps the library's own compiled model), a default schedule, and `default_sampler()`. It cannot be stepped from Python; only its sampler runs it.

Here is an illustrative sketch. The library calls and capability values are placeholders, not nutpie's actual API.

```python
class NutpieSampler(Sampler):
    name = "nutpie"
    capabilities = SamplerCapabilities(
        controls=frozenset({"pause", "resume", "cancel"}), progress="per_chunk",
        streaming_output=True, custom_schedule=False, live_stats=True, runs_pymc_steps=False)

    def check_supports(self, step):
        if not isinstance(step, NutpieNUTS):
            raise UnsupportedStep(...)

    def start(self, spec, storage, monitor):
        tune, draws = tune_and_draws(spec.schedule)            # rejects anything but (warmup, sample)
        run = start_library_run(spec.bundle.compiled_model, tune=tune, draws=draws,
                                chains=len(spec.states), seed=..., blocking=False)
        return ExternalRunHandle(run, translate=to_draw_batches, storage=storage, monitor=monitor)
```

### Entry points: plugging in without changing PyMC

An entry point is standard Python packaging metadata. A plugin package declares, in its own `pyproject.toml`:

```toml
[project.entry-points."pymc.samplers"]
nutpie = "nutpie.pymc_adapter:NutpieSampler"

[project.entry-points."pymc.steps"]
nutpie-nuts = "nutpie.pymc_adapter:NutpieNUTS"
```

PyMC finds installed plugins with `importlib.metadata.entry_points(group="pymc.samplers")`, by name:

- **Nothing is imported until someone asks for that name.** Discovery reads metadata only.
- **Passing an instance always works too.** `sampler=NutpieSampler(...)` needs no registration.
- **What this buys:**
  - PyMC stops hard-coding the list of external samplers.
  - Version checks and output patching (`_nutpie_meets_min_version`, `patch_nutpie_idata`) move out of `pymc/sampling/mcmc.py` and into the adapters.
  - Anyone can publish a step or a sampler that `pymc.sample` runs, without a PR to PyMC.

### Versioning the extension API

The dependency is on **the contract**, not on PyMC releases.

- PyMC exposes a separate integer, `pymc.sampling.api.API_VERSION`. It is bumped only when the extension contracts change, with a deprecation window. The same idea is used by `__array_api_version__` and DLPack versions.
- The adapter module imports PyMC lazily, so the plugin library still works without PyMC installed. It reads `API_VERSION` and picks the matching adapter, so introspect-and-dispatch works and there is no permanent dependency:

  ```python
  api = pymc.sampling.api.API_VERSION
  NutpieSampler = _SamplerV2 if api >= 2 else _SamplerV1
  ```
- The library's CI runs PyMC's conformance kit against each PyMC version it supports.
- **Where adapters live:**
  - In the library: recommended, because whoever knows the library maintains its adapter.
  - In PyMC: as today.
  - In a third package: also possible.

### Security: protecting entry points from hijacking

**Threat model.**

- Any installed distribution can declare any entry-point name, including `nutpie`. A typosquatted package, or a compromised transitive dependency, could therefore register `nutpie` and take over `pymc.sample(sampler="nutpie")`.
- That is the risk worth defending against.
- An unfamiliar name such as `"jibberish"` is a different case. The user typed it, so asking for it is the user's trust decision.

**What PyMC can't prevent, and why.** A malicious package that is already installed can run code without going through entry points at all:

- at build time, for sdists;
- at every interpreter start, through a `.pth` file;
- whenever it is imported.

From there it could monkeypatch `pymc` directly.

So entry-point checks can't make a compromised environment safe. What they can do:

- make sure a name means what the user thinks it means;
- make collisions loud;
- make provenance visible.

The integrity of the environment as a whole depends on other tools: lockfiles with hashes, trusted publishing and attestations on the package index, and dependency review.

**Failing at install time isn't possible.** Installing a wheel runs no code, and PyMC can't hook into the installation of another package. The earliest point at which PyMC sees a plugin is when it reads installed metadata.

**The resolver.** It applies these rules:

```python
BUILTIN = {"pymc": ProcessSampler, "process": ProcessSampler, "inline": InlineSampler}
RESERVED = {                         # well-known names → the distributions allowed to provide them
    "nutpie": {"nutpie"},
    "numpyro": {"numpyro"},
    "blackjax": {"blackjax"},
}

def resolve_sampler(name: str) -> Sampler:
    if name in BUILTIN:                                            # 1. built-ins can't be shadowed
        return BUILTIN[name]()
    eps = entry_points(group="pymc.samplers").select(name=name)
    if not eps:
        raise UnknownSampler(name, available=list_samplers())
    providers = {normalize(ep.dist.name) for ep in eps}
    if len(eps) > 1:                                               # 2. duplicate names are an error
        raise AmbiguousSampler(name, providers)                    #    the user passes an instance instead
    (ep,) = eps
    expected = RESERVED.get(name)
    if expected is not None and normalize(ep.dist.name) not in expected:
        raise UntrustedProvider(name, provider=ep.dist.name, expected=expected)   # 3. reserved names are pinned
    if not plugin_policy().allows(name, ep.dist.name):             # 4. optional allowlist
        raise PluginNotAllowed(name, ep.dist.name)
    cls = ep.load()                                                # 5. plugin code is imported only now
    log.info("Sampler %r from %s %s (%s)", name, ep.dist.name, ep.dist.version, ep.value)
    return cls()
```

The resolver is backed by three more measures:

- **Nothing loads automatically.**
  - `import pymc` never scans or loads plugins.
  - A default `pymc.sample()` never consults entry points.
  - PyMC never chooses a plugin by itself (no "use nutpie if it's installed").

  This is the difference from systems that auto-load plugins, such as pytest's `pytest11`.
- **Provenance is recorded.** The output's attributes record the distribution name, its version and the object path. This also helps reproducibility.
- **A check command for CI:** `python -m pymc.plugins check` lists every plugin with its provider and fails on collisions or reserved-name violations. The first registry access in a session also warns about reserved-name violations, even for names nobody requested.

**Trusted providers.** Yes, in two forms:

- **Reserved names**, enforced: PyMC pins the rightful owner of the names it knows about, and a mismatch is a hard error.
- **An opt-in allowlist** for locked-down environments, set through configuration or an environment variable such as `PYMC_PLUGINS_ALLOW=nutpie,numpyro`, plus `PYMC_DISABLE_PLUGINS=1`.

I'd advise against warning by default on every plugin that isn't on a PyMC-maintained list:

- it produces warning fatigue;
- it implies PyMC vouches for code it doesn't review;
- it makes PyMC a gatekeeper for the ecosystem, which works against the first goal of this refactor.

## Conformance kit

### How it is generic

Every `Step` has the same signature, and `init` and `transition` are pure functions of `(state, bundle)`. So the kit can drive any step directly, with no sampler and no processes. The precedent is scikit-learn's `check_estimator` and `parametrize_with_checks`.

```python
from pymc.testing.conformance import parametrize_with_checks, check_sampler

@parametrize_with_checks([RandomWalk(), GHMC(damping=0.9)])
def test_conformance(step, check):
    check(step)

def test_nutpie_sampler():
    check_sampler(NutpieSampler, step=NutpieNUTS())
```

- Each check is a function `check_x(obj, ctx)`. The kit selects the checks that apply from what the object under test declares: its type (`Step`, `PopulationStep` or `Sampler`), its `requirements`, its `stats` and its `Params`.
- Reference models live in `pymc.testing.reference_models`. Each pairs a model with a way to draw *exact* samples from its posterior:
  - iid normal;
  - a correlated normal with condition number 10³;
  - a bounded variable with a transform;
  - a discrete variable;
  - a mixed model;
  - a target for population steps.

  The kit picks models according to `competence` and `requirements`.
- With fixed seeds the statistical checks aren't flaky: a given seed either always passes or always fails. Tolerances are calibrated to Monte Carlo standard error so that false failures are very rare.

### The checks

| Check | How |
|---|---|
| Contract and types | `init` and `transition` return the declared `State`; the returned stats match the declared names, dtypes and shapes; the step, state and bundle pickle. |
| Purity | Snapshot the state, then `transition` the original and the snapshot: the outputs are identical, and an unrelated snapshot is unchanged. |
| Resumability | Pickle the state mid-run and continue: the result is bitwise identical to an uninterrupted run. This follows from purity. |
| **Invariance** (the important one) | With params frozen, start M chains (e.g. 2,000) from **exact** draws of the reference posterior, plus exact auxiliary state such as GHMC's momentum. Run k steps. The M end points must still follow the posterior: per-coordinate KS tests and tests of mean and covariance, Bonferroni-corrected. It needs no convergence, so it is cheap and catches most bugs. |
| Joint distribution test | Geweke's "getting it right" test: on a model whose data can be simulated, alternate between drawing data given θ and applying the step to θ given the data. θ's marginal must match the prior. |
| Adapters | Pure; frozen in non-adaptive phases; a metric adapter recovers a known covariance on a Gaussian. For `validity="diminishing"`, the parameter changes shrink as declared. |
| Population (SPMD) | Runs with 1, 2 and 3 shards agree; replicated state is identical on every shard. |
| Composition | Invariance still holds inside a compound step alongside a reference step. |
| Loop-level samplers | The controls listed in `capabilities` work (pause stops progress; cancel leaves valid partial output); events arrive in order; translated batches validate against `DrawBatch`; draw counts match. |
| Calibration (optional, heavy) | SBC. |

The kit builds on `StepMethodTester` in `tests/helpers.py`, which already round-trips `sampling_state`.

## Lowering the loop to an accelerated language (future)

The design allows the canonical loop to run in JAX or numba, provided three conditions hold:

1. **The state is pure.** This design already guarantees it, with counter-based RNG keys replacing `np.random.Generator` at that point.
2. **The step's `transition` and the adapters can be expressed in the target language.** That holds when `transition` is itself a PyTensor graph, which could then live *in the bundle* as a compiled `step_k` that scans over K steps, or when the step provides a lowered implementation.
3. **Side effects happen only at chunk boundaries:** K compiled steps, then one `DrawBatch` back in Python, followed by monitor events and control checks.

This is why `SamplerCapabilities.progress` has a `"per_chunk"` value, and why steps are not allowed to call back into storage or monitors from inside `transition`.

## Open questions

- The names `Step` and `Sampler`, or the alternative `Sampler` and `Executor` (see [Naming: `Step` and `Sampler`](#naming-step-and-sampler-open-for-discussion)).
- Where `pm.NUTS`, `pm.Metropolis` and the other public names point once new steps exist: the legacy classes until a deprecation, or the new steps, with the legacy classes moved to a legacy module.
- The Storage and Monitor contracts, to be settled in M3, using the loop-level external samplers as the stress test.
- The public/internal split of modules.
- Whether any existing `pymc.sample` argument should eventually be deprecated (for example `nuts_sampler` in favour of `sampler`), or all of them stay indefinitely.
- The default schedule of a compound step: how to align phase boundaries across blocks.
- Extending a finished run (more draws) as opposed to resuming a stopped one.
- Bitwise reproducibility across shard counts (allgather-then-reduce versus allreduce of partial results).
- Mixing chain and population steps in one compound (shelved).
- A thread-based sampler, which needs `clone()` and free-threaded PyTensor.
