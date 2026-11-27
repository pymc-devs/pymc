# Roadmap

This document specifies the planned architecture for refactoring `pymc.sample` around a small set of public, typed components:

- steps, with their states and compiled bundles;
- initialization strategies;
- adaptation schedules;
- samplers, which run the steps;
- the output path to storage and monitors.

It is ordered top down: the implementation plan, what the refactor gains (including the naming of its two central abstractions), how `pymc.sample` uses the components, and then each component in turn, from steps down to cross-chain communication and plugins.

**Status:** design draft. All code is pseudocode.

## Overview

### Discussed in person

In the in person meeting we had agreed on a very crude restructure of what we wanted from `pymc.sample`. I'll copy over verbatim and then try to summarize it into what I interpret are the main points:

```python
# Instantiate a Sampler object
sampler = pm.nuts.sample(target_accept=0.9, draws=200)  # Not this
pm.nuts(target_accept=0.9).sample(draws=200)  # This. There should be no extra **kwargs in sample

# ---
# Migrate sampler arguments into sampler and away from sample
pm.sample(draws=200, target_accept=0.9)   # TODAY
pm.sample(sampler=pm.nuts(target_accept=0.9), draws=200)  # Migrate towards this

# ---
# Backends and argument forwarding
pm.sample_prior_predictive(backend="jax")
pm.sample(backend="jax")
pm.sample_posterior_predictive(backend="jax")

# Maybe also a special blocking behavior?
def sample(
    *,
    model=None,
    sampler=None,
    random_seed,
    tune,
    init: InitialStrategy,
    draws,
    chains,
    cores,
    blas_cores,
    monitor = ProgressBar(
        initial_status_update=True,
        final_status_update=True,
    ),
    backend: str,
    storage = DataTreeStorage(
        var_names=None,
        discard_tune=True,
    ),
    blocking: bool = True,
):
    if sampler is None:
        sampler = pm.NUTS or pm.StepSampler(...)
    future = sampler.sample(**kwargs)
    if blocking:
        return await future
    else:
        return future

# ---
# how do we resume sampling

with pm.Model():
  ...

  # Not this
  sampler = pm.nuts()
  tuned_sampler = sampler.tune(chains=4)
  draws = tuned_sampler.resume()

# ---

  # Not this
  sampler = pm.nuts()
  dt, tuned_sampler = pm.sample(tune=500, sampler=sampler, return_sampler_state=True)
  pm.sample(sampler=tuned_sampler)

# ---

  # This, we later decide if we need some return_sampler_state kwarg or if't over-engineered
  # By default all samplers encode the last-state meta-data they need to resume sampling
  dt = pm.sample()
  sampler = pm.sampler_from_dt(dt)
  pm.sample(sampler=sampler)  # does tune=None become tune=0

# ---
# Storage
pm.sample(
    idata_kwargs =  # Not this
)

posterior = pm.sample( # This
    storage=pm.trace.DataTree(**idata_kwargs),
    storage=pm.trace.Zarr(**zarr_kwargs),
)

# ---
# Logging and monitoring
MyProtocol:
  def monitor(self, chain_idx, trace, step_state):
      logging.logger()  # or whatever
      ...

pm.sample(monitor="progressbar" | MonitorProtcol):
```

### Intended goals

- Create a `Sampler` object that is capable of running the MCMC sampling process
- Forward arguments into `Sampler` creation so that `Sampler.sample` then uses no extra `**kwargs` and accepts almost the same signature as `pymc.sample`
- Potentially enable blocking and non blocking sampling
- Enable ability to resume sampling from a partial trace that is held in some kind of storage.
- Define a `Storage` contract that enables the previous point, and have the `Storage` initialization outside of `sample`.
- I suppose, that `pymc.sample` will still have to create a default `Storage` when the user provides none.
- Add a `Monitor` protocol that can be used as the progress monitor. This could be the progress bar that we have or something else.

## High level overview of my proposal

I want to try and tackle all of these points and in order to do so, I tried to go into the potential consequences for different implementation choices.

- **`Sampler` objects**: currently we have no formal `Sampler` contract. We have serial sampling, parallel sampling, SMC and then external samplers, all in `pymc.sampling`. All implement their own protocol of how draws are created, logged and written to storage. The ideal implementation would house all of these under shared contracts and make it easier for external libraries to add their own samplers to be used.
- **`Step` objects**: currently, we have some `Step` methods, like the built in `NUTS`, but then these get conflated with external libraries running NUTS variants (like nutpie or blackjax). If we want to create `Sampler` objects with a uniform contract, we will also have to create a new `Step` object contract. Just to be clear, the `Step` objects define the kernels that do the MCMC transitions.
- **Adaptation and initialization**: we currently have some `init` that are `Step` specific, and hard to make compatible with other samplers. Ideally, we should separate the adaptation, and initialization strategies from the `Step` objects, so that custom init-adapt patterns can be used with ease. There will still be limitations for external samplers, and for `Step` objects that can handle a subset of adaptation methods, but this might be cleaner than subclassing `Potential`s, as needs to be done today.
- **Storage and Monitor**: these objects will have to work with the `Sampler` objects because they control the MCMC flow of information, and they will have to be responsible for writting and logging. That means that `Storage` and `Monitor` should also expose a consistent API for use.
- **Non blocking and asynchronous behavior**: external samplers might have non blocking sampling, but our builtin ones do not. If we want to enable this functionality we will have to think through how the `Sampler` will instantiate concurrency, background work and communication. The simplest case will be independent chain sampling (e.g. NUTS), where each chain could run in isolation. The next case would be population sampling (e.g. SMC or `DEMetropolis` like step methods) where chains can communicate information with each other at each step and take steps as a whole. These two kinds of sampling will have to be distinguished and reconciled under a uniform API.
- **Compiled bundles separate from stateless `Step` objects**: our current `Step` objects are linked to the model object and compiled `logp` or `dlogp` callables, making them stateful and tightly linked to models. Ideally, we would have the `Step` methods be stateless, have the step method specific attribute values travel in a separate `State` object, and then have another object that holds all the compiled model objects. This separation will make it easier for external libraries to conform to the API and just implement the pieces that they need.
- **Resuming sampling**: If the `Step` methods are stateless and only need an external `State` object and compiled bundle, loading these from some `Storage` unit will enable the `Sampler` to resume from a previous place. We will have to reconcile this with `Adapatation` and `Initialization` phases if they are separate objects, but that's something doable.
- **Storage**: Still need to plan their API better, but they should expose some methods to be able to work in tandem with the new `Sampler` objects. The requirement is that all `Storage` objects should be capable of storing some representation of the `Sampler`'s `State` object so that resuming sampling is possible.
- **Monitor**: Still need to plan their API better, but they should expose some methods to be able to work in tandem with the new `Sampler` objects. The requirement is that there needs to be at least one monitor that can work in non-interactive systems.

## Concrete plan and internal milestones

The milestones are ordered by dependency, and each one leaves `main` releasable. **`pymc.sample` stays blocking and synchronous throughout, and keeps returning its results as it does today.** Most of its current arguments are absorbed by components. They keep working, with a deprecation warning, until they are removed; see [Backwards compatibility](#backwards-compatibility).

| Milestone | Scope | Depends on | Exit criteria |
|---|---|---|---|
| **M0** Contracts | A `pymc.sampling.api` module with the dataclasses and ABCs in this document, and `API_VERSION = 0`. No behaviour change. | — | The types import, and this document matches them. |
| **M1** Canonical loop | `InlineSampler`; the phase and schedule machinery; a `LegacyStep` adapter so every existing `BlockedStep` runs in the new loop; conformance kit v0 (contract, purity, resumability, invariance). | M0 | All existing step methods sample through `InlineSampler` via `LegacyStep`. A ported random-walk Metropolis passes the kit. |
| **M2** HMC split *(optional)* | NUTS and HMC ported as stateless steps. `DualAveraging`, `WindowedDiag`, `WindowedFull`, … extracted from `QuadPotential*`. `InitStrategy`, plus translation of the `init=` strings. | M1 | Posterior regression tests match legacy NUTS. Every legacy `init` string maps to an initialization plus NUTS's `mass_matrix_adapter`. |
| **M3** Output and monitoring | `DrawBatch` and bundle `postprocess`. The `Storage` and `Monitor` base classes and their contracts, finalized here. Two storages: in-memory (`DataTreeStorage`, the default) and zarr (`ZarrStorage`); both persist the sampler state. Monitors: a progress bar matching today's, a `FileMonitor` that is the default in non-interactive sessions, and the adapter that keeps `callback=` working during its deprecation. | M1 | Steps run through `LegacyStep` produce the same `DataTree` on the old and new output paths. Both storages pass the kit's storage checks, including the checkpoint round trip. Without a terminal (a batch job), the default monitor writes progress to a file. |
| **M4** Process sampler and non-blocking API | `ProcessSampler` with free-running workers; `RunHandle` with pause, resume, stop, cancel and checkpoint; resume from storage, including after a crash; `start_sampling` and `sample_async`. `pymc.sample` is routed through the new stack: still blocking, still returning a `DataTree`, with the superseded arguments translated under a deprecation warning. | M3; M2 if included | The test suite passes on the new stack. With legacy steps and the same seed, the new stack reproduces today's draws. A run resumed after `stop()`, or after its process was killed, is bitwise identical to an uninterrupted one (same machine and backend). |
| **M5** Plugins and external samplers | The entry-point registry with its security rules; adapters for nutpie, numpyro and blackjax, each providing its library's `Step`s and `Sampler`; `nuts_sampler=` deprecated in favour of `sampler=`; sampler conformance checks. | M3, M4 | The deprecated `nuts_sampler="nutpie"` and `sampler="nutpie"` produce the same run. The three adapters pass the kit. |
| **M6** Population tier | `Comm` (inline and process), consistent cuts, SMC, ChEES, and DEMetropolis moved off `PopulationStepper`. | M4 | SMC and DEMetropolis results match legacy. Runs with 1, 2 and 3 shards agree. |
| **M7** API version bump | Consolidate the classes and API, and bump `API_VERSION = 1`. | All previous | Have internal agreement on the API. |
| *Not included, but made possible for the future* | A thread-based sampler (free-threaded Python); loops lowered to JAX or numba; mixing chain and population steps in one compound; steps that change the model during a run; automatic, model-tuned step assignment. | — | — |

**If M2 is left out**, NUTS and HMC keep their adaptation inside `QuadPotential*` and run through `LegacyStep`. `LegacyStep` maps the end of the warmup phase to today's `stop_tuning()`. The `init=` strings keep going through today's `init_nuts`. Everything else still works, because M3–M6 don't depend on M2. What waits for M2 is composable adaptation for HMC: custom schedules and adapters, and `InitStrategy` objects passed to `init=`.

## Rationale

The first, **M1**, most important thing that we need to do is to establish an internal consensus on where we are going and how we plan to get there. That's why I propose starting by defining the APIs for the new classes as contracts that future implementation must uphold. If we find that the API was limiting in some way, we can extend it. That's why **M1** sets out with `API_VERSION = 0` and **M7** finally consolidates into the final API that works and we are satisfied with. Potential future changes will have to bump the `API_VERSION` further up. At the end of the phase, we have no extra functionality.

The next thing, **M2**, that I propose is to have a working `Sampler` that is not asynchronous nor runs in the background with multiple processes, and then build `Step`s, their `State`s, compiled bundles, and simple `Storage`, and `Monitor`s that use that. We will also write a set of tests that will measure how much of the planned contract is covered by the classes. At the end of the phase, we have no extra functionality.

After that, we start with the heavy work of **M3**:

- Wrapping porting step methods
- Splitting adaptation and initialization
- Adapting the backends to use the new `Storage` protocol
- Adapt the `Monitor` classes to conform with the protocol

At the end of the phase, we still have not exposed the new internal API to `sample` and can't show any added functionality

Then **M4** builds support for the more sophisticated `Sampler` loops: concurrency of workers (like what we have with `pymc.sampling.parallel` now), enabling communication between the main process and the worker processes, enable pausing, non blocking and resuming sampling. This is the phase at which we bring our new stack in to `pymc.sample` and the pause, resume, non blocking sampling will become available for internal samplers that don't rely on population steps.

**M5** wraps the external samplers and defines the way that external libraries register their sampler into `pymc`. We can coordinate with `nutpie` and maybe `blackjax` about how they can register their entry points, but in practice, at the end of the phase, the external samplers should be made available to `sample` through a simpler interface.

**M6** ties the last piece of MCMC with population samplers and a cross chain communication protocol. So SMC should become integrated into `sampling` at the end of this milestone.

## What we gain

### 1. An extension path that needs no change to PyMC

A developer can write either of two things in their own package:

- a `Step`: one transition, which PyMC's samplers run;
- a `Sampler`: a whole sampling loop, such as an external library's.

They declare an entry point in their own package like:

```toml
[project.entry-points."pymc.samplers"]
my_package = "my_package.pymc_adapter:MyPackageSampler"

[project.entry-points."pymc.steps"]
my_package-step = "my_package.pymc_adapter:MyPackageStep"
```

and `pymc.sample(..., sampler="my_package")` runs it (or is able to choose its step).

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

### 3. Validated arguments

Each step declares:

- the typed state it carries.
- the compiled bundle it needs.
- the parameters it exposes for tuning.
- the space it works in: transformed or not, gradient-based or not, continuous or discrete.

Steps and samplers take their configuration as typed constructor arguments, and **a setting that the chosen step or sampler can't use raises**. It is never ignored, and never only warned about. Examples: `cores=4` with `InlineSampler`, a custom schedule with a sampler that runs its own warmup, `backend="jax"` with a sampler that only compiles with numba.

A mismatch is an error at setup, not NaNs or divergences halfway through a run.

### 4. A `pymc.sample` that stops growing

New functionality arrives as components (`step`, `init`, `schedule`, `sampler`, `storage`, `monitor`) instead of more keyword arguments. The existing arguments that a component supersedes are deprecated: during the deprecation period they are translated into components with a warning naming the replacement, and then they are removed. What stays is a small set of settings that every step and sampler shares, such as `draws`, `chains` and `random_seed`.

### 5. Non-blocking sampling, with `pymc.sample` still blocking

`pymc.sample` keeps its current behaviour: it blocks until sampling finishes and returns the same kind of result as today, a `DataTree`.

The non-blocking alternative is `start_sampling`. It takes the same arguments and immediately returns a `RunHandle`, which offers progress, pause, resume, graceful stop, cancel and checkpoint. `sample_async` awaits the same handle.

One handle type covers both PyMC's samplers and external ones, and each sampler declares which controls it supports.

### 6. Better monitoring, and resumable sampling

Step states are explicit, serializable and self-contained, RNG included. Every storage persists them, so a run can be resumed later: after a deliberate stop, from a returned `DataTree`, or after a crash, from the last checkpoint in durable storage.

Monitors receive typed events and declared statistics instead of per-step callbacks. Progress display, logging and early-stopping rules (for example, stop at an ESS target) all become pluggable. In a non-interactive session, such as a batch job, the default monitor writes progress to a file instead of drawing a progress bar nobody sees.

### 7. Cross-chain methods as first-class citizens

ChEES-style pooled adaptation, SMC, ensemble samplers and tempering share one population contract with explicit collective communication. This replaces both the $O(\text{chains}^2)$ broadcast in `PopulationStepper` and the separate SMC process machinery.

### Out of scope for now

- Thread-based samplers. v1 parallelism is process-based; see [Threads: not supported in v1](#threads-not-supported-in-v1).
- Compound steps that mix chain-level and population-level steps.
- Loops lowered to JAX or numba. The design leaves room for them; see the last section.
- Steps that change the model during a run. The contracts don't rule them out; see [Models that change during a run](#models-that-change-during-a-run).
- **Automatic, model-tuned step assignment.** This would be an expert system that recognizes patterns in a model (conjugate priors, rewritable structure, blocks with a specialized Gibbs update) and builds a step tuned to that model. Users compose steps by hand, as they do today. Such a system would sit where `assign_steps` sits and return an ordinary `CompoundStep`, so nothing downstream would change.

## How the pieces fit

```
                            user code
                                │  pymc.sample (blocking) / start_sampling / sample_async
                                ▼
┌──────────────────────── setup, in the calling process ─────────────────────────┐
│ arguments ──► components (Step, InitStrategy, Schedule, Sampler, Storage, ...) │
│ Model ──model_transform──► CompiledBundles: the Step's and each init stage's   │
│ Schedule = [Phase(adapters, until, warmup), ...]   validated against the Step  │
│ init pipeline (root >> stages) ──draw──► InitResult: positions + Approximation │
│ Step.default_params(bundle, InitResult) ──► one StepState per chain            │
└───────────────────────────────────┬────────────────────────────────────────────┘
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
| `InitStrategy` | One stage of the initialization pipeline, which yields the starting positions and an optional approximation of the posterior | PyMC, users |
| `Adapter` | Tunes one parameter field from the step's signals | PyMC, users |
| `Phase`, `Schedule` | Which adapters run, for how long, and whether the draws count as warmup | PyMC (defaults), users |
| `StopCondition` | When a phase ends: iterations, ESS, wall-clock time | PyMC, users |
| `Sampler` | Runs the loop: inline, in processes, or inside an external library | PyMC; extension authors |
| `RunHandle` | The user's view of a running sampler | sampler |
| `DrawBatch` | The common output format every sampler emits | sampler |
| `Storage`, `Monitor` | Where draws and checkpoints go; how progress is reported | PyMC, users |

## `pymc.sample`

### Backwards compatibility

**What stays the same**

- **`pymc.sample` is blocking and synchronous.** It returns when sampling has finished, with no event loop or handle involved, and Ctrl+C stops the run as it does today.
- **It returns the same kind of result as today:** a `DataTree`, produced by the default storage. Whether it should instead return the run's `Storage` object, which converts to a `DataTree`, is decided when storage is prototyped in M3 (see [Open questions](#open-questions)). Either way, every storage subclasses one `Storage` base class and meets one contract, so code that consumes the result doesn't depend on which storage produced it.

**What changes**

- **Arguments that a component supersedes are deprecated.** Examples: `nuts_sampler` (now `sampler`), `trace` and `idata_kwargs` (now `storage`), `callback` (now `monitor`), and step keyword arguments such as `target_accept` (now the step's constructor). [What happens to each existing argument](#what-happens-to-each-existing-argument) lists them all.
- **During the deprecation period**, one translation function turns them into components and emits a `FutureWarning` naming the replacement. After that they are removed.
- **`return_inferencedata=False` and `MultiTrace` go away** together with the arguments that select them.

**What is checked**

- **Adding the new names can't break existing code.** Today an unrecognised keyword ends up in `step_kwargs`, and `assign_step_methods` raises `ValueError` for it (`pymc/sampling/mcmc.py:239-250`). So no working code passes any of the new names.
- **Disagreement is an error.** Passing an argument together with the component that supersedes it fails when they disagree, for example `cores=4` with `sampler=ProcessSampler(cores=2)`.
- **A setting the chosen step or sampler can't use raises** (see [Validated arguments](#3-validated-arguments)). The translation function decides once which component each setting goes to, so a sampler's `check_supports` only has to reject what that sampler can't honour.

### Signature

```python
def sample(
    draws: int = 1000,
    *,
    tune: int | None = None,                   # None → each step's default, as today
    chains: int | None = None,
    cores: int | None = None,                  # shorthand for the default ProcessSampler(cores=...)
    random_seed: RandomState = None,
    progressbar: bool = True,                  # shorthand: False selects a silent monitor
    model: Model | None = None,
    step: Step | BlockedStep | Sequence[Step | BlockedStep] | None = None,  # legacy steps run via LegacyStep
    sampler: Sampler | str | None = None,
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    init: InitStrategy | InitPipeline | None = None,   # None → step.default_init(); strings are deprecated
    schedule: Schedule | None = None,
    storage: Storage | None = None,
    monitor: Monitor | Sequence[Monitor] | None = None,
    model_transform: Callable[[Model], Model] | None = None,
    backend: str | None = None,                # PyTensor backend of the compiled bundle
    compile_kwargs: dict | None = None,
    **deprecated,                              # superseded arguments: translated with a FutureWarning;
                                               # any other name raises TypeError
) -> DataTree:                                 # or the run's Storage; decided in M3
    return start_sampling(...same arguments...).result()     # blocks; re-raises errors; Ctrl+C → stop()


def start_sampling(...same arguments...) -> RunHandle:          # the non-blocking alternative
    ...


async def sample_async(...same arguments...) -> DataTree:
    return await start_sampling(...same arguments...).wait_async()
```

- **`sample` is exactly `start_sampling(...).result()`.** There is one implementation, so the blocking and non-blocking paths can't drift apart.
- **`draws` and `tune` only feed the default schedule, `default_schedule(step, tune, draws)`.** `tune=None` resolves to the step's `default_tune`. A compound step takes the largest of its blocks', as `get_default_tune_steps` does today (`pymc/sampling/mcmc.py:128`).
- **Settings that only feed the default schedule can't be combined with an explicit one.** Passing a `schedule` together with `draws` or `tune` is an error. So is passing it together with a step that sets an adaptation setting, such as `pm.NUTS(target_accept=0.9)` (see [Steps](#the-contract)). They are never silently ignored.
- **`backend` and `compile_kwargs` become compile options once**, through `resolve_backend_compile_kwargs` (`pymc/pytensorf.py:918`) as today, and go to `Bundle.build`. An external sampler that compiles the model itself declares which backends it accepts, and raises for any other.
- **`cores` and `progressbar` are kept as shorthands** because nearly every call uses them. Whether they stay permanently is an open question.

### Body

```python
def start_sampling(draws=1000, *, model=None, step=None, sampler=None, init="auto", schedule=None,
                   storage=None, monitor=None, model_transform=None, **arguments) -> RunHandle:
    model = modelcontext(model)

    # 0. Turn shorthands and deprecated arguments (with a FutureWarning) into components;
    #    raise on conflicts with explicit components
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
    sampler.check_supports(step, c.settings)      # raises for a step or a shared setting it can't use

    # 3. Schedule, validated before anything that takes a while: compilation, then
    #    initialization (ADVI, Pathfinder). It needs the step and the init strategy, not the bundle.
    init = with_root(c.init or step.default_init(), initvals=c.initvals)   # SupportPoint() first if missing
    if c.schedule is None:
        schedule = default_schedule(step, tune=c.tune, draws=c.draws)
    else:
        reject_adaptation_settings(step)       # e.g. NUTS(target_accept=0.9), in any block
        schedule = c.schedule
    check_schedule(schedule, step, init, sampler.capabilities)  # targets, signals, blocks, validity, hints
    schedule = resolve_blocks(schedule, step)   # each entry → (its block's position, or None; adapter)

    # 4. Compile once, in this process: the step's bundle and one per initialization stage,
    #    all from the same transformed model, through one cache
    bundles = BundleCache(model, transform=model_transform, compile_kwargs=c.compile_kwargs)
    bundle = bundles.get(step.Bundle, step.vars)               # backend= already folded in
    check_space(step.requirements, bundle.space)

    # 5. Initialization, before any state exists
    chains = c.chains or sampler.default_chains()
    rngs = get_random_generator(c.random_seed).spawn(chains)   # one stream per chain, as today
    seeds = [rng.integers(2**30) for rng in rngs]              # as today; the chains continue on rngs
    dist = None
    for stage in init.stages:                                  # stages[0] is the root
        dist = stage(bundles.get(stage.Bundle, step.vars), dist, seed=seeds[0])
    init_result = InitResult(positions=dist.draw(seeds), approximation=dist.approximation)
    params = step.default_params(bundle, init_result)
    states = {ch: step.State.new(position=init_result.positions[ch], params=params, rng=rngs[ch])
              for ch in range(chains)}

    # 6. Output, then start
    storage = c.storage or default_storage(model, bundle.space)
    monitor = c.monitor or default_monitor(sampler.capabilities)
    spec = RunSpec(step=step, bundle=bundle, states=states, schedule=schedule, hint=init_result)
    storage.begin(spec)
    return sampler.start(spec, storage, monitor)
```

**Seeds are derived exactly as today** (`pymc/sampling/mcmc.py:921-922`). Each chain's generator first gives one integer seed, and initialization uses only those seeds. The final draw gives chain i the i-th seed. Stages that run once for all chains, such as ADVI, use the first chain's seed, as `init_nuts` does today. Each chain then continues on its own generator. This is what lets legacy steps on the new stack reproduce today's draws for the same seed, which is an exit criterion of M4.

The flow is the same for PyMC's steps and for external libraries. An external library provides its own `Step`s and its own `Sampler`, under the same two contracts, and keeps them separate: the step carries the algorithm's settings, and the sampler the loop's. For example, `NutpieNUTS(target_accept=0.9)` is a step that only `NutpieSampler` can run, so its `default_sampler()` returns a `NutpieSampler`; see [Plugins: external steps and samplers](#plugins-external-steps-and-samplers). So `pm.sample(step=NutpieNUTS())`, `pm.sample(sampler="nutpie")` and the deprecated `pm.sample(nuts_sampler="nutpie")` all resolve to the same run.

### What happens to each existing argument

| Existing argument | Status | Becomes |
|---|---|---|
| `draws`, `tune` | kept | the default schedule, `default_schedule(step, tune, draws)` |
| `chains`, `random_seed`, `model`, `backend`, `compile_kwargs` | kept | settings shared by every step and sampler |
| `initvals` | kept | the root of the initialization pipeline: `SupportPoint(initvals=...)` |
| `step` | kept | the `Step`; legacy `BlockedStep`s are wrapped in `LegacyStep` |
| `cores` | kept, as a shorthand | `ProcessSampler(cores=...)` by default; any other sampler must declare that it honours `cores`, or it raises |
| `progressbar` | kept, as a shorthand | `False` selects a silent monitor |
| `init` | kept; strings deprecated | an `InitStrategy` or a pipeline of them; a string becomes an initialization plus NUTS's `mass_matrix_adapter` (table below) |
| `**kwargs` for steps (`target_accept`, `nuts={...}`) | deprecated | the step's constructor: `step=pm.NUTS(target_accept=0.9)` |
| `n_init`, `jitter_max_retries` | deprecated | the init stages' arguments: `ADVI(n=...)`, `SupportPoint(jitter=True, max_retries=...)` |
| `nuts_sampler`, `nuts_sampler_kwargs` | deprecated | `sampler="nutpie"`; the adapter splits `nuts_sampler_kwargs` between its step (`NutpieNUTS(...)`) and its sampler (`NutpieSampler(...)`) |
| `mp_ctx`, `blas_cores` | deprecated | `ProcessSampler(mp_ctx=..., blas_cores=...)` |
| `progressbar_theme`, `quiet`, `callback` | deprecated | monitors: `ProgressBarMonitor(theme=...)`, `CallbackMonitor(fn)`, … |
| `compute_convergence_checks` | deprecated | whether the monitors' final report includes convergence checks |
| `trace`, `var_names`, `discard_tuned_samples`, `keep_warning_stat`, `idata_kwargs` | deprecated | the storage: `ZarrStorage(...)` in place of `trace=ZarrTrace(...)`, otherwise `DataTreeStorage(var_names=..., ...)` |
| `return_inferencedata` | deprecated, then removed along with `MultiTrace` | — |

The translation has a few known pitfalls, each of which gets its own test: array-like `random_seed` values, `tune=None` resolving to each step's default, and forwarding `backend` to external samplers.

With M2, each deprecated `init` string becomes an initialization and a value for NUTS's `mass_matrix_adapter` adaptation setting, as in the table below. NUTS's default schedule always also has `DualAveraging` for the step size. The adapters are shown without their bindings, which come from the presets in `hmc` (see [Adapters](#adapters)). Without M2, the strings keep going through today's `init_nuts` until they are removed.

| `init=` today | Initialization | NUTS's `mass_matrix_adapter` |
|---|---|---|
| `"adapt_diag"` | `SupportPoint()` | `WindowedDiag()` (today's foreground/background window scheme) |
| `"jitter+adapt_diag"` (`"auto"`) | `SupportPoint(jitter=True)` | `WindowedDiag()` |
| `"jitter+adapt_diag_grad"` | `SupportPoint(jitter=True)` | `WindowedDiagGrad()` (today's `QuadPotentialDiagAdaptExp`) |
| `"advi+adapt_diag"` | `SupportPoint() >> ADVI(n=200_000)` | `WindowedDiag(from_hint=True)`, seeded from the ADVI approximation |
| `"advi"`, `"advi_map"` | `SupportPoint() >> ADVI()`, `SupportPoint() >> MAP() >> ADVI()` | `False`: a fixed diagonal inverse mass matrix from the hint |
| `"map"`, `"jitter+map"` | `SupportPoint() >> MAP()`, `SupportPoint(jitter=True) >> MAP()` | `False`: a fixed dense one, the inverse of the Hessian at the MAP |
| `"adapt_full"`, `"jitter+adapt_full"` | `SupportPoint()`, `SupportPoint(jitter=True)` | `WindowedFull()` |

These reproduce today's starting points, with one deliberate difference. Today's ADVI-based inits ignore `initvals`: `pm.fit` starts from the model's default initial point (`pymc/variational/opvi.py:777`), and the starting points computed before it are discarded. In the new pipelines ADVI starts from the root, so it uses `initvals` when they are given.

### Blocking, non-blocking and resuming

```python
# As today: blocks until done and returns a DataTree
idata = pm.sample(1000, tune=1000, chains=4, step=pm.NUTS(target_accept=0.9))

# The non-blocking alternative: same arguments, returns immediately
handle = pm.start_sampling(1000, chains=4, storage=ZarrStorage("run.zarr", checkpoint_every=100))
handle.progress()        # per chain: iteration, phase, selected stats
handle.pause(); handle.resume()
handle.stop()            # graceful: stops at a consistent point, keeps draws, saves states
idata = handle.result()  # blocks; re-raises worker errors

# From a coroutine, e.g. in a Jupyter cell or a web server
idata = await pm.sample_async(1000, chains=4)

# Later, possibly in a new Python process, even after a crash: continue from the last checkpoint
idata = pm.sample(storage=ZarrStorage("run.zarr"))

# The returned DataTree carries the sampler state too, so a stopped run continues from its result,
# also after it was saved to disk and loaded again
idata = pm.sample(storage=DataTreeStorage.from_datatree(idata))
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

- **Existing steps keep working unchanged** through `LegacyStep` (M1), including user-defined and pymc-extras steps, while they are ported.
- **Steps are ported one at a time:** random-walk Metropolis in M1, then NUTS and HMC in the optional M2.
- **The numerical cores are reused, not rewritten:** proposal distributions, `metrop_select`, the leapfrog integrator and NUTS tree building, the slice logic, `competence`, and the `stats_dtypes_shapes` declarations, which become `stats`. The one structural change inside them is that the integrator receives the inverse mass matrix as a parameter instead of holding a stateful potential.
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
    adaptation_settings: ClassVar[tuple[str, ...]] = ()   # settings that only configure default_adapters
    default_tune: int = 1000                 # tuning iterations when tune=None

    def __init__(self, vars=None, **config): ...         # immutable configuration only

    # Setup
    @classmethod
    def competence(cls, var, has_grad) -> Competence: ...
    def default_params(self, bundle: BundleT, hint: InitResult) -> ParamsT: ...
    def default_init(self) -> InitStrategy | InitPipeline: ...
    def default_adapters(self) -> tuple[Adapter | OnBlock, ...]: ...   # bound to this step's own fields
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
    sampler: str | None = None          # set when only one sampler can run this step, e.g. "nutpie"
```

**Rules**

1. The step object holds configuration only (`target_accept`, `max_treedepth`, …). It never caches anything computed during sampling. It must be picklable and cheap to copy. That makes it a coordinator, not a container of state.
   - Configuration is typed: one constructor argument per setting, named as the algorithm's library names it.
   - A setting, or a combination of settings, that the step can't use raises in the constructor.
   - **Adaptation settings**, such as NUTS's `target_accept`, only configure the step's default adapters. The step lists them in `adaptation_settings`. Each defaults to `None`, which keeps the adapter's own default. A step that sets one can't be combined with an explicit schedule: setup raises, as it does for `tune` and `draws`.
2. `transition` is a pure function of `(state, bundle)`; the one exception is advancing the state's RNG (see [RNG and ownership](#rng-and-ownership)).
3. **A step doesn't know whether it is tuning.** It only sees `state.params`. Whether those params are still being adapted is the adapters' business, and whether draws count as warmup is the phase's.
4. `init` fills the caches derived from the position (log density, gradient). It is idempotent, so it serves fresh and resumed states alike.
5. The step never touches storage, monitors, processes or other chains. Statistics leave as return values, and the sampler routes them.
6. `stats` declares every statistic with its dtype and shape, so storage can preallocate and monitors know what is available.
7. **A step provides its default adapters, not its default schedule.** `default_adapters` binds adapters to the step's own fields, usually from PyMC's presets (see [Adapters](#adapters)). PyMC puts them in a tuning phase followed by a sampling phase (see [Phases and schedules](#phases-and-schedules)). So every default schedule has the same structure, and the blocks of a compound step always line up. A richer default, such as Stan's windows, is a schedule the user passes.

### Example: random-walk Metropolis

This is everything an author writes for a simple step.

```python
class RWParams(Param):
    scale: ParamField[np.ndarray]             # (ndim,)

class RWSignals(Signal):
    accept_prob: SignalField[float]

@dataclass(frozen=True)
class RWState(StepState[RWParams]):
    logp: float | None = None                 # cache, filled by init()


class RandomWalk(Step[RWState, LogpBundle, RWParams, RWSignals]):
    State, Bundle, Params, Signals = RWState, LogpBundle, RWParams, RWSignals
    stats = {"accepted": Stat(np.bool_),
             "accept_prob": Stat(np.float64, arviz_name="acceptance_rate", monitor=True)}
    requirements = Requirements(space="any", gradient=False, values="continuous")

    def default_params(self, bundle, hint):
        approx = hint.approximation
        return RWParams(scale=np.sqrt(approx.cov.diag()) if approx else np.ones(bundle.space.ndim))

    def default_init(self):
        return SupportPoint(jitter=True)

    def default_adapters(self):
        return (AcceptanceScale(RWParams.scale, accept=RWSignals.accept_prob, target_rate=0.234),)

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
    Phase(
        "tune",
        (
            AcceptanceScale(
                RWParams.scale,
                accept=RWSignals.accept_prob,
                target_rate=0.44,
            ),
        ),
        Iterations(500)
    ),
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
  - Its state is a tuple of sub-states, one per block, and so are its signals.
  - Its bundle is a tuple of sub-bundles, all built from the same transformed model.
  - Its `transition` is a Gibbs sweep: before each block steps, that block receives the other blocks' current values in `state.cond`.
- Changing `cond` invalidates the block's cached log density, so the compound runs that block's `init` before its `transition`. This is no more expensive than today: legacy step methods recompute the log density at the start of every step.
- External steps, such as a blackjax kernel wrapped as a `Step`, compose the same way. When blocks use different array libraries, `cond` is converted at block boundaries (zero-copy via DLPack where possible).
- Out of scope for v1: mixing chain steps and population steps in one compound.

**Tuning a compound step.** An adapter is bound to the fields of a step *type*, such as `HMCParams.step_size`. That doesn't say which block it tunes when two blocks share a type. So in a compound's schedule, each adapter is also put on a block:

```python
nuts, rw = pm.NUTS(vars=[mu, sigma]), RandomWalk(vars=[tau])
schedule = [
    Phase("tune", (replace(hmc.step_size, target_accept=0.9).on(nuts),
                   hmc.diag_mass_matrix.on(nuts),
                   AcceptanceScale(RWParams.scale, accept=RWSignals.accept_prob).on(rw)),
          Iterations(1000)),
    Phase("sample", (), Iterations(1000), warmup=False),
]
pm.sample(step=CompoundStep((nuts, rw)), schedule=schedule)
```

- **A block is named by its step object.** For a compound you didn't build, such as one from `assign_steps`, `step.block_of("mu")` returns the block that samples `mu`. Every variable is in exactly one block, so the lookup is unambiguous.
- **`adapter.on(block)` returns an `OnBlock`**, which checks the binding against the block's type as soon as it is written (see [Adapters](#adapters)). The adapter itself is unchanged, so one preset can go on several blocks.
- **In a compound's schedule, every adapter is on a block.** An adapter on its own only works with a single step.
- **At setup, each block becomes its position.** `check_schedule` checks that every block an `OnBlock` names belongs to the step being sampled. Then the entry is replaced by the block's position in the compound. While sampling, the loop only uses positions: each adapter reads its block's entry of the compound's signals, and its values go into its block's hyperparameters (see [The canonical loop](#the-canonical-loop)). Adapters never see the compound's signals, and the loop never sees step objects.
- **The default schedule merges the blocks' defaults.** A compound's `default_adapters` returns each block's default adapters, put on that block, and its `default_tune` is the largest of its blocks'. Every default schedule is a tuning phase followed by a sampling phase, so the blocks always line up.
- **Adaptation settings are checked in every block.** `CompoundStep((pm.NUTS(target_accept=0.9), rw))` with an explicit schedule raises.

### Population steps

```python
class PopulationStep(Step[PopStateT, BundleT, ParamsT, SignalsT]):
    shardable: ClassVar[bool] = False        # may members be split across workers?

    def transition(self, state: PopStateT, bundle: BundleT, comm: Comm
                   ) -> tuple[PopStateT, Stats, SignalsT]: ...


@dataclass(frozen=True)
class PopulationState(Generic[ReplicatedT, MembersT]):
    replicated: ReplicatedT    # identical on every shard: SMC's beta and stage, a pooled mass matrix, a shared RNG
    members: MembersT          # leading axis = the members this shard holds: positions, log densities, weights
```

The only difference from a chain step is the `comm` argument. It gives access to collective operations over the whole population (see [Cross-chain communication](#cross-chain-communication)). A population step running unsharded on one worker receives an identity `comm`.

### Continuous adaptation: GHMC, MEADS and adaptive Metropolis

The previous draft required the final phase to have no adapters. That rule is too strict. Three different things get called "continuous adaptation", and the framework puts each one in a different place.

| Kind | Example | Where it lives | Why the draws are valid |
|---|---|---|---|
| **Part of the Markov state** | GHMC's partially refreshed momentum, $p \leftarrow \alpha p + \sqrt{1-\alpha^2}\,\xi$ (Horowitz, 1991) | `GHMCState.momentum`; the damping α in `Params` | The kernel is fixed on the extended space $(x, p)$ and leaves $\pi(x)\,\mathcal{N}(p)$ invariant |
| **Ensemble kernel** | MEADS (Hoffman & Sountsov, 2022): each fold's step size, damping and preconditioner come from the *other* folds' current states. Also stretch moves. | A `PopulationStep`; the "adapted" quantities are computed inside `transition` | It's a fixed kernel on the product space. Each fold's update conditions only on the other folds, so it leaves $\prod_i \pi(x_i)$ invariant. Updating every fold at once from each other's states would not. |
| **History-dependent adaptation** | Adaptive Metropolis (Haario et al., 2001); covariance estimators whose updates shrink like $1/n$ | An `Adapter` declaring `validity="diminishing"` | Not invariance: ergodicity holds under diminishing adaptation and containment (Roberts & Rosenthal, 2007) |

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


class HMCParams(Param):                     # see "Parameters and signals"
    step_size: ParamField[float]
    inverse_mass_matrix: ParamField[Covariance]


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

**Hidden RNGs.** Compiled graphs can contain shared random-generator variables with default updates. A step's `init` reseeds the ones in its bundle from the state's generator, as `reseed_rngs` does today. The initial-point function, whose jitter uses them, belongs to initialization and is reseeded from each chain's seed (see [Initialization](#initialization)). No randomness may come from anywhere else.

### Serialization, checkpoints and resume

- Every state must round-trip through `pickle`.
- A checkpoint holds four things: the step states, the position in the schedule (phase index and progress within the phase), the adapter states, and the `InitResult`, which adapters still read when a later phase starts. Without the adapter states, a run stopped during warmup couldn't resume.
- **Every storage persists checkpoints.** This is part of the `Storage` contract, not a feature of some backends (see [Storage](#storage)).
  - Durable storage, such as zarr, writes them as the run goes, so a run can be resumed after a crash, from its last checkpoint. The `_sampling_state` group of `ZarrTrace` already exists for this.
  - In-memory storage keeps the latest checkpoint and puts it in a group of the returned `DataTree`, so a stopped run can be resumed from its result, including after the result has been saved to disk.
- Resuming means loading a checkpoint, discarding any draws stored after it, building a fresh bundle, and calling `init` on each state. Initialization doesn't run again. Because the RNG is part of the state, the resumed draws are the ones an uninterrupted run would have produced.
- The bundle fingerprint stored in the state detects a changed model or changed data. In that case resume fails, unless the user explicitly accepts the change. The fingerprint is compared only when resuming, never during a run.
- For population steps, the checkpoint holds the whole `PopulationState`, captured at a consistent cut.
- **Loading a checkpoint unpickles it**, so resuming from a file runs code from that file. Only files from trusted sources should be resumed. A structured encoding (arrays, plus JSON for parameters and the RNG state) can later replace pickle for PyMC's own states.

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

    def postprocess(self, batch: DrawBatch) -> DrawBatch: ...   # unravel, untransform, deterministics
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

- `build` runs in the calling process. It applies `model_transform` first, so static changes to the model happen once, before any worker starts. Examples: `remove_value_transforms`, `do`, `observe`, freezing data, reparameterizations.
- The PyTensor backend (C, numba, JAX) is chosen at this point, through `pymc.sample`'s `backend=` or `compile_kwargs`. PyMC's steps don't take a `backend` argument.
- Each bundle is pickled once per worker. Unpickling rebuilds the compiled functions, at a cost that depends on the backend:
  - C loads from its compile cache, which is cheap.
  - numba may re-JIT unless its cache is warm.
  - JAX re-traces on the first call.
  - With the `fork` start method, workers inherit the compiled functions and pay nothing.
- PyMC already works this way today: step methods that hold compiled functions are pickled to each worker (`_Process._unpickle_step_method`).

### Rules for the callables' arguments

1. Every per-call input is an explicit argument: the position, other blocks' values (`cond`), β, a batch of particles. This replaces how `ArrayStepShared` conditions on other blocks today, through `shared[var].set_value(...)`, which is hidden mutable state.
2. In PyMC's bundles, model data is fixed for the lifetime of the bundle. Changing data means building a new bundle; the fingerprint changes, and `init` recomputes the caches. This is a property of PyMC's bundles, not a requirement of the contract; see the next section.
3. Randomness enters only through the state's RNG.
4. Callables never cross a process boundary on every step. The bundle travels once; states travel at the start and at checkpoints; each step sends only small records.

### Models that change during a run

No step or sampler that PyMC ships changes the model while it samples. Trans-dimensional samplers (reversible jump, birth–death), samplers that add or remove mixture components, and samplers that rewrite the model as they learn about it are not planned. **The contracts must not rule them out**, though, so that someone else can build one without fighting the framework.

Such a step stays stateless, as follows:

- **The bundle may hold the `pymc.Model` itself** and compile from it on demand, keeping compiled functions in a cache keyed by the model's current structure.
- **What changes goes in the state**: which components exist, the current dimension. It reaches the bundle's callables as an explicit argument, just as `cond` and β do.
- The state remains the only thing that evolves. Memoized compilation inside the bundle can't be observed from outside it, so `transition` is still a pure function of `(state, bundle)`.

The contracts promise the following, so that such an extension keeps working:

- **Samplers treat a bundle as opaque**, apart from `space`, `fingerprint`, `postprocess`, `clone` and pickling. They never assume that `functions` is fixed, or that it was compiled before the run started.
- **The fingerprint identifies the bundle's starting model**, and is compared only on resume.
- **Shapes may vary.** A state's `position` has no fixed length in the contract; `ModelSpace` describes the starting point; `Stat` shapes can have `None` dimensions. `postprocess` receives the whole `DrawBatch`, so a bundle can read the current structure from the batch's stats.
- **A storage that can't store varying shapes raises** (`StorageCapabilities.variable_shapes`), instead of truncating or padding.

### Threads: not supported in v1

PyTensor `Function`s aren't reentrant. Each one owns its input and output storage and the containers of the shared variables it updates, RNGs included. Two threads calling the same function overwrite each other's values.

Processes avoid the problem, because each worker unpickles its own copy. So the v1 position is: **parallelism is process-based, and thread-based samplers aren't supported yet.**

The contract still includes `clone()`, with a default implementation that calls `Function.copy()` on every field, and it is samplers that decide when to clone, not step authors. A thread-based sampler, useful with free-threaded Python, can then be added later without changing any step or bundle.

## Initialization

Initialization runs in the calling process, after the bundles are built and before any step state exists. It is a pipeline of stages:

1. A **root** stage gives the model's support point.
2. Each **later stage** starts from what the previous one returned and refines it: MAP, ADVI, Pathfinder, jitter.
3. Each chain's starting position is **drawn** from what the last stage returned.

```python
@dataclass(frozen=True)
class Approximation:                       # Gaussian, in bundle.space
    mean: np.ndarray
    cov: Covariance                        # DiagCov, DenseCov or LowRankPlusDiag; see "Parameters and signals"


class InitialDistribution(ABC):            # what a stage returns, and the next stage starts from
    space: ModelSpace
    approximation: Approximation | None    # a Gaussian summary, when the stage has one

    @abstractmethod
    def draw(self, seeds: Sequence[int]) -> np.ndarray: ...    # one point per seed: (len(seeds), ndim)


class InitStrategy(ABC):                   # one stage
    Bundle: ClassVar[type[CompiledBundle]]
    root: ClassVar[bool] = False                          # True for SupportPoint and PriorDraw, False otherwise
    provides_approximation: ClassVar[bool] = False        # True for ADVI, Pathfinder, Laplace

    @abstractmethod
    def __call__(self,
        bundle,
        upstream: InitialDistribution | None,    # None only for a root
        seed: int,
    ) -> InitialDistribution: ...

    def __rshift__(self, other: "InitStrategy") -> "InitPipeline": ...


@dataclass(frozen=True)
class InitPipeline:                        # stage >> stage >> ...
    stages: tuple[InitStrategy, ...]       # stages[0] is a root
    def __rshift__(self, other: InitStrategy) -> "InitPipeline": ...


@dataclass(frozen=True)
class InitResult:                          # the "hint" that the step and the adapters receive
    positions: np.ndarray                  # (chains, ndim), in bundle.space
    approximation: Approximation | None = None


@dataclass(frozen=True)
class ModelBundle(CompiledBundle):         # for stages that compile their own graphs
    model: Model                           # after model_transform
    compile_kwargs: dict
```

### The root: today's initial point

- `SupportPoint(initvals=None, jitter=False, max_retries=10)` is today's initial point: each variable's support point or its `initval`, overridden by `initvals`, in the step's space.
  - With `jitter=True`, it adds today's uniform jitter inside the compiled graph, and redraws a point whose log density isn't finite, up to `max_retries` times. This is `_init_jitter` unchanged, so it reproduces today's starting points.
  - `initvals` can be one dict or one per chain, as today. With one per chain, the i-th point drawn uses the i-th chain's values, and drawing more points than there are chains raises.
- `PriorDraw()` draws from the prior instead, in the step's space. SMC starts from it, and SMC raises at setup for any other pipeline, because its tempering assumes the particles come from the prior. `PriorDraw()` itself raises at setup for a model whose prior can't be drawn from, for example one with a `Flat` variable.
- Both compile from the model, because `initvals` and per-variable choices such as `initval="prior"` change the graph. Their `Bundle` is `ModelBundle`.
- A pipeline that doesn't start with a root gets `SupportPoint()` put in front of it. `pymc.sample(initvals=...)` sets that root's `initvals`; passing them both to `pymc.sample` and to an explicit root raises.

### Later stages

- Each one receives the previous stage's `InitialDistribution`, and draws as many starting points from it as it needs: one for MAP and ADVI, one per path for Pathfinder.
- It returns a new distribution: a point mass at the mode for MAP, a Gaussian for ADVI, the importance-resampled draws for Pathfinder.
- That is why stages pass distributions and not an array with one position per chain: how many points a stage needs has nothing to do with the number of chains.
- `Jitter(scale, max_retries)` adds uniform noise in numpy to the draws of the stage before it, and redraws a point whose log density isn't finite. Its `Bundle` is `LogpBundle`, and it passes the approximation on, as in `Pathfinder(num_paths=4) >> Jitter(0.1)`. To reproduce today's draws, jitter the root instead (`SupportPoint(jitter=True)`).
- The other stages are `MAP()`, `ADVI(n)`, `Pathfinder(num_paths)`, or a short run of another step.

### How stages reach the model

- A stage declares its `Bundle`, as a step does. Every bundle, the step's and each stage's, is built from the same transformed model, through one build cache. A callable that several bundles need, such as the log density and gradient that both NUTS and Pathfinder use, compiles once.
- So what a stage returns is always in the space the step samples in. A stage holding the untransformed model couldn't guarantee that: under a `model_transform` that marginalizes a variable, it would still optimize that variable.
- A stage that needs the model itself declares `ModelBundle`. A native Pathfinder declares a bundle with `logp_dlogp` and a batched log density. A wrapper around the current pymc-extras implementation, which takes a model, can declare `ModelBundle` until it is ported.
- Initialization runs in the calling process, so the stages' bundles never travel to workers.

### Results stay out of the bundles

The bundles are a function of the model, the transform and the backend, while initialization results depend on the seed. Keeping the two apart keeps the fingerprint stable for resume, lets a bundle be reused for another run of the same model, and keeps per-chain data out of what is sent to workers. The `InitResult` belongs to the run, and checkpoints save it.

What initialization learns can still shape the run, in two ways that don't involve the bundles:

- **If it should change what is compiled**, for example choosing between a centered and a non-centered parametrization from a quick fit, it is a `model_transform` that happens to run inference, and it runs before any bundle is built.
- **If it should change how the step moves**, for example whitening with the approximation's covariance, it goes into the step's parameters, such as NUTS's inverse mass matrix, and nothing is recompiled.

### Drawing the starting positions

- The last stage's distribution is drawn once, in the calling process, with one seed per chain, and the workers receive the positions. Drawing them all together matters:
  - adaptation can need every chain's start: today, `adapt_diag` starts its running estimate from their mean (`pymc/sampling/mcmc.py:1893`);
  - Pathfinder's resampling should draw without replacement across chains;
  - a distribution can hold compiled functions or JAX state, which no worker needs.
- **Seeds:** initialization only uses the integer seeds derived from the chains' generators (see [Body](#body)). The final draw gives chain i the i-th seed. The stages that run once for all chains use the first chain's seed, as `init_nuts` does today. A stage that needs more randomness spawns child seeds from its own.
- The seeds are integers, not generators, because today's initial-point function is reseeded from an integer (`reseed_rngs`), and that is what reproduces today's draws.

### Hand-off to adaptation

- The `InitResult` is the hint that `step.default_params(bundle, hint)` and `adapter.init(current, hint)` read. Some adapters use the positions: today's `"jitter+adapt_diag"` starts its running estimate from their mean. Others use the approximation: today's `"advi+adapt_diag"` becomes `SupportPoint() >> ADVI()` plus `WindowedDiag(from_hint=True)`.
- **Declared output:** each stage declares whether it provides an approximation of the posterior or only starting positions. A pipeline provides one if any stage does and the later stages pass it on, as `Jitter` does. `check_schedule` uses the declaration: an adapter that needs the approximation is rejected at setup when initialization can't provide one. For example, `init=SupportPoint(jitter=True)` with `WindowedDiag(from_hint=True)` fails with "SupportPoint provides starting points only; WindowedDiag(from_hint=True) needs an approximation, for example from ADVI(), Pathfinder() or MAP()". This follows the `returns_posterior` flag of Jesse Grabowski's [inference-configuration sketch](https://gist.github.com/jessegrabowski/9918bb7b4cd7cc6243a39183b826b71f).

## Adaptation and schedules

### How tuning is organized

A step has several hyperparameters: NUTS has a step size and an inverse mass matrix, random-walk Metropolis a proposal scale. During warmup, tuning sets some of them from the chain's own draws.

- **One adapter per tuned hyperparameter.** Each tuned hyperparameter is handled by its own `Adapter` object. After every iteration, the adapter reads what the step reported about that iteration (its *signals*, such as the acceptance statistic), updates its estimate, and gives the hyperparameter a new value.
- **Adapters that run together form a `Phase`.** A phase is a stretch of iterations that lasts until its stop condition holds, and its adapters are applied after every iteration of that stretch. A phase without adapters keeps every hyperparameter fixed.
- **A sequence of phases is a `Schedule`.** Tuning can take several phases, as in Stan's windows, or a diagonal estimate followed by a dense one. The last phase of a run usually has no adapters, and holds the draws that are kept.

Today's default NUTS run, in these terms:

```python
schedule = [
    Phase(
        "tune",
        (
            DualAveraging(
                HMCParams.step_size,
                accept=HMCSignals.accept_stat,
                target_accept=0.8,
            ),
            WindowedDiag(
                HMCParams.inverse_mass_matrix, draws=HMCSignals.position
            ),
        ),
        until=Iterations(1000),
    ),
    Phase("sample", (), until=Iterations(1000), warmup=False),
]
```

1. Before the first tuning iteration, each adapter starts from its hyperparameter's initial value, set by `step.default_params`, and from the initialization result.
2. In each tuning iteration, the step moves the chain and reports its signals. Each adapter reads the signals it is bound to, updates its estimate, and proposes a new value for its hyperparameter. The next iteration uses the new values.
3. When the phase ends, each adapter hands over a final value. Dual averaging, for example, explores with a noisy step size and hands over a smoothed average.
4. The sampling phase has no adapters, so the hyperparameters stay fixed, and its draws are kept.

The step never knows whether it is being tuned: it only reads its hyperparameters from its state. The rest of this section describes each piece in turn.

### Parameters and signals

A step is coupled to adaptation through exactly two declarations: what it lets adapters tune (`Params`) and what it reports to them after each iteration (`Signals`).

```python
class HMCParams(Param):                            # tunable
    step_size: ParamField[float]
    inverse_mass_matrix: ParamField[Covariance]    # the momentum is drawn from N(0, M); this is M⁻¹

class HMCSignals(Signal):                          # offered to adapters after each iteration; not stored by default
    accept_stat: SignalField[float]
    position: SignalField[np.ndarray]
    grad: SignalField[np.ndarray]
```

- **Fields are named as the step's algorithm names them.** No adapter expects a particular name (see [Adapters](#adapters)).
- **A subclass of `Param` or `Signal` is a frozen dataclass, without the decorator.** Each field's type is wrapped in `ParamField` or `SignalField`. A default is given with `param_field(default=...)` or `signal_field(default=...)`, which take the arguments of `dataclasses.field`.
- **A field accessed on the class gives a reference to it.** `HMCParams.step_size` is a `ParamRef[HMCParams, float]`, and `HMCSignals.accept_stat` is a `SignalRef[HMCSignals, float]`. On an instance, the same attribute gives the value, a `float`.
- **References are typed.** Type checkers know both types, so a binding to the wrong kind of field is flagged where the schedule is written. A reference is an attribute, not a string: an editor can find or rename every use of a field, and a typo raises `AttributeError` on the line that builds the schedule.

**The inverse mass matrix is a covariance.** HMC draws its momentum from N(0, M), and its kinetic energy is ½ pᵀM⁻¹p. Tuning sets M⁻¹ to an estimate of the posterior covariance. That is why the parameter is the inverse, and why its type is `Covariance`, the same type initialization uses for its approximation:

```python
class Covariance(ABC):                     # DiagCov, DenseCov, LowRankPlusDiag, ...
    def __matmul__(self, v: np.ndarray) -> np.ndarray: ...              # Σ v
    def diag(self) -> np.ndarray: ...
    def draw(self, rng: np.random.Generator) -> np.ndarray: ...         # x ~ N(0, Σ)
    def draw_inverse(self, rng: np.random.Generator) -> np.ndarray: ... # x ~ N(0, Σ⁻¹)
```

- HMC uses `draw_inverse` for the momentum, and `@` for the velocity M⁻¹p and the kinetic energy. These are the operations of today's `QuadPotential` (`random`, `velocity`, `energy`) without the adaptation, which moves to the adapters. Today's `QuadPotentialDiag` and `QuadPotentialFull` are already built from a variance and a covariance.
- `Covariance` is an ordinary base class with concrete subclasses. A new structure, such as block-diagonal or Kronecker, is a new subclass plus the adapter that estimates it, with no change to the step.
- Because the type describes what is estimated, not how HMC uses it, the same estimators serve other steps: adaptive Metropolis binds `WelfordDense` to its proposal covariance. `default_params` can also start from the initialization's `Approximation.cov` as it is.
- A learned nonlinear transport map, as in nutpie's normalizing-flow adaptation, isn't a covariance: it changes the coordinates the log density is evaluated in. A step that supports one declares a separate parameter for it, with its own type and adapter. PyMC's NUTS doesn't in v1; nutpie's step does.

**How references work.** `Param` turns each subclass into a frozen dataclass, then puts a descriptor on the class for each field. The descriptor answers class access with a reference, and stays out of the way on instances. `Signal`, `SignalField`, `signal_field` and `SignalRef` are the same, for signals.

```python
@dataclass(frozen=True)
class ParamRef(Generic[O, T]):
    owner: type[O]                                 # the class the reference was taken from
    name: str                                      # the field
    def get(self, params: O) -> T:
        return getattr(params, self.name)


class ParamField(Generic[T]):
    def __init__(self, name: str):
        self.name = name

    @overload
    def __get__(self, obj: None, owner: type[O]) -> ParamRef[O, T]: ...
    @overload
    def __get__(self, obj: O, owner: type[O]) -> T: ...
    def __get__(self, obj, owner):
        return ParamRef(owner, self.name)          # only reached on class access, see below

    if TYPE_CHECKING:
        def __set__(self, obj: object, value: T) -> None: ...   # tells type checkers that __init__ takes a T


def param_field(*, default=MISSING, **kwargs) -> Any:
    return dataclasses.field(default=default, **kwargs)


@dataclass_transform(frozen_default=True, field_specifiers=(param_field,))
class Param:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in inspect.get_annotations(cls):  # a re-declared inherited field must not
            if name not in cls.__dict__:           # take its parent's descriptor as its default
                setattr(cls, name, dataclasses.field())
        dataclass(frozen=True)(cls)                # dataclass must see plain fields only
        for f in dataclasses.fields(cls):
            setattr(cls, f.name, ParamField(f.name))

    @classmethod
    def ref_to(cls, name: str) -> ParamRef[Self, Any]: ...   # raises on an unknown name
```

- **The dataclass comes first, the descriptors after.** A descriptor present while `dataclass` runs is taken for the field's default: dataclass would call `ParamField.__get__(None, cls)` and make the resulting `ParamRef` the default. That is why subclasses don't use the `@dataclass` decorator; `@dataclass(frozen=True)` on a subclass raises `TypeError`.
- **`dataclass_transform` only informs type checkers.** It tells them that subclasses of `Param` are frozen dataclasses, and that `param_field` declares a field. At run time it only records its arguments on the class.
- **Class access gives the reference; instance access never reaches the descriptor.** `HMCParams.step_size` calls `__get__(None, HMCParams)`. The generated `__init__` stores each value in the instance's `__dict__`, and a descriptor with only `__get__` ranks below the instance's `__dict__`, so `params.step_size` is an ordinary attribute read. This matters for signals, which are built after every iteration. Reads and construction cost about the same as for a plain frozen dataclass: 15 instead of 10 ns per read, and 146 instead of 134 ns to build a three-field signal.
- **`__set__` exists only for type checkers.** The typing spec takes the type of a descriptor-typed field's `__init__` parameter from `__set__`. Without it, checkers expect `__init__` to receive a `ParamField[float]`. At run time, a `__set__` would make every instance read go through `__get__`: 50 ns per read, and 275 ns to build a three-field signal. So type checkers see a data descriptor and the interpreter a non-data one. They agree on everything observable: `__init__` takes a `T`, an instance gives a `T`, the class gives a reference, and assignment fails because the dataclass is frozen.
- **References taken through a subclass.** `NUTSParams.step_size` and `HMCParams.step_size` have different owners but name the same hyperparameter. Once `check_schedule` has confirmed that a reference's owner is the step's `Params` or a class it inherits from, a hyperparameter is identified by its name. mypy 1.19 types an inherited field accessed through a subclass as its value (`float`), not as a reference; ty types it correctly. So bindings go through the class that declares the field.
- **`ref_to(name)`** builds a reference from a name known only at run time, for example in a loop or from a configuration file. Its value type is `Any`, so only `check_schedule` checks it.
- **Checked with mypy 1.19 and ty 0.0.86.** Both infer `ParamRef[HMCParams, float]` for `HMCParams.step_size` and `float` for `params.step_size`. Both flag a wrong `__init__` argument, a misspelt field, an assignment, and a step-size adapter bound to the covariance. `dataclasses.replace` and pickling work as usual. A bare default (`max_treedepth: ParamField[int] = 10`) works at run time, but both checkers reject it, hence `param_field`.
- This is the typing pattern of SQLAlchemy 2.0's `Mapped[int]`, where `User.id` is a column expression and `user.id` an int.

### Adapters

```python
@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True)
class Adapter(ABC, Generic[ValueT, InputsT, AdaptStateT]):
    validity: ClassVar[Literal["warmup_only", "diminishing"]] = "warmup_only"

    param: ParamRef[Any, ValueT]               # the hyperparameter it tunes; ValueT is that field's type

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True, kw_only=True)(cls)   # a subclass's own fields are keyword-only

    @property
    def needs_approximation(self) -> bool:     # needs hint.approximation from initialization
        return False

    @abstractmethod
    def read(self, signals: Signal) -> InputsT: ...     # picks its inputs out of the step's signals

    def init(self, current: ValueT, hint: InitResult | None) -> AdaptStateT: ...
    def update(self, a: AdaptStateT, inputs: InputsT, comm: Comm) -> AdaptStateT: ...
    def propose(self, a: AdaptStateT) -> ValueT: ...    # the value the step uses in the next iteration
    def final(self, a: AdaptStateT) -> ValueT: ...      # the value the hyperparameter keeps after the phase

    def on(self, block: Step) -> OnBlock:               # puts it on one block of a compound step
        return OnBlock(block, self)
```

```python
DualAveraging(HMCParams.step_size, accept=HMCSignals.accept_stat, target_accept=0.8)
WelfordDiag(HMCParams.inverse_mass_matrix, draws=HMCSignals.position)

# The same algorithms, bound to other steps' hyperparameters
DualAveraging(MetropolisParams.scaling, accept=MetropolisSignals.accept_rate, target_accept=0.234)
WelfordDense(AdaptiveMetropolisParams.proposal_cov, draws=AdaptiveMetropolisSignals.position)

# Each adapter declares its references as typed fields, and reads its inputs through them
class DualAveraging(Adapter[float, float, DualAveragingState]):   # one input, so InputsT is float
    accept: SignalRef[Any, float]
    target_accept: float = 0.8

    def read(self, signals: Signal) -> float:
        return self.accept.get(signals)

class DrawsAndGrads(NamedTuple):
    draws: np.ndarray
    grads: np.ndarray

class FisherDiag(Adapter[Covariance, DrawsAndGrads, FisherDiagState]):
    draws: SignalRef[Any, np.ndarray]
    grads: SignalRef[Any, np.ndarray]

    def read(self, signals: Signal) -> DrawsAndGrads:
        return DrawsAndGrads(self.draws.get(signals), self.grads.get(signals))
```

- **An adapter is bound when it is constructed.** Its first argument says which hyperparameter it tunes. Each of its inputs gets a keyword argument that says which signal feeds it.
- **Subclasses become frozen dataclasses without a decorator**, as with `Param`.
  - `__init_subclass__` makes each subclass a frozen dataclass whose own fields are keyword-only. `Adapter` itself is an ordinary dataclass, so `param` stays positional, and a subclass of a subclass works the same way.
  - `dataclass_transform` tells type checkers the same thing. mypy and ty both see `param` as positional and the other arguments as keywords, and both flag a missing `param`, a signal passed by position, and an assignment to a field.
  - Decorating a subclass with `@dataclass(frozen=True)` as well raises `TypeError`.
- **Each reference is a field of its own, with its own type.** A reference held in a container, such as a dict of signals, would lose its type, and `check_schedule` couldn't find it.
- **Its code never sees the step's field names.** `read` picks the adapter's inputs out of the signals through its references. It returns them as `InputsT`: the value itself for a single input (a `float` for `DualAveraging`), and a small named tuple for several. The loop passes the result to `update`. That is how one `DualAveraging` tunes NUTS's step size or a Metropolis step's scaling, and how each step can keep its algorithm's names.
- **`InputsT` types `update` in each adapter.** Inside `FisherDiag.update`, `inputs.grads` is an `np.ndarray`. It has to be a type parameter: a subclass that narrowed a base `update(inputs: Mapping[str, Any])` to `DrawsAndGrads` would break substitutability, and mypy and ty both reject that override. For the same reason, `read` takes any `Signal`: an adapter doesn't know which step it serves.
- **Types only check a binding, they never choose a field.** Two `float` fields can't be confused, because the binding names one of them. The check happens twice:
  - When the schedule is written, a type checker flags a binding to the wrong kind of field, such as `DualAveraging` on `inverse_mass_matrix`, or fed from `position`.
  - At setup, `check_schedule` checks the same thing again, for users who don't run a type checker and for references built with `ref_to`. An adapter object doesn't carry its type parameters, so `check_schedule` reads them from its class. The annotations say what each reference must point to (`accept: SignalRef[Any, float]`), and the class's generic arguments give `ValueT` (`Adapter[float, float, DualAveragingState]`).
  - Both apply the same rule: the field must have the type the adapter declares. So `WelfordDiag`, an `Adapter[Covariance, ...]`, can't tune a field declared as `DiagCov`, although its `final` returns one. No planned step declares a covariance that narrowly.
- **A step's `default_adapters` binds PyMC's adapters to its own fields**, so most users never write a binding. A custom schedule writes one per adapter, or starts from the presets below. An adapter holds configuration only, so the same adapter object can appear in several phases; each phase starts it with a fresh `AdaptState`.
- **`validity` and `needs_approximation` describe the algorithm.** `validity` is fixed for a class. `needs_approximation` is a property, because it can depend on the configuration: for example, if `ShrunkDense` could shrink toward the initialization's approximation instead of the current value.
- **Few people write adapters.** Each one writes a `read` of a line or two, and one with several inputs also writes a named tuple for them.
- Adapters are pure, like steps. `AdaptState` is serializable and saved in checkpoints.
- `comm` lets an adapter pool its estimate across chains, for example one inverse mass matrix shared by all chains. For independent adaptation it is the identity.

**Presets.** Next to each step, PyMC provides adapters already bound to the step's fields, with every setting at its default. The step's `default_adapters` takes them from there, and applies its adaptation settings:

```python
# pymc/step_methods/hmc/adapters.py, imported as hmc
step_size = DualAveraging(HMCParams.step_size, accept=HMCSignals.accept_stat)
diag_mass_matrix = WindowedDiag(HMCParams.inverse_mass_matrix, draws=HMCSignals.position)
dense_mass_matrix = WindowedFull(HMCParams.inverse_mass_matrix, draws=HMCSignals.position)

replace(hmc.step_size, target_accept=0.9)      # dataclasses.replace: the preset, with one setting changed


class NUTS(Step[...]):
    adaptation_settings = ("target_accept", "mass_matrix_adapter")

    def __init__(self, vars=None, *, target_accept: float | None = None,
                 mass_matrix_adapter: Adapter | Literal[False] | None = None, max_treedepth=10, ...): ...

    def default_adapters(self):
        step_size = hmc.step_size
        if self.target_accept is not None:
            step_size = replace(step_size, target_accept=self.target_accept)
        if self.mass_matrix_adapter is False:     # the inverse mass matrix stays at its default_params value
            return (step_size,)
        return (step_size, self.mass_matrix_adapter or hmc.diag_mass_matrix)
```

- **Users change a default through the step**, for example `pm.NUTS(target_accept=0.9)` or `pm.NUTS(mass_matrix_adapter=hmc.dense_mass_matrix)`. A custom schedule uses the presets directly.
- **A preset is an adapter**, so it goes straight into a phase. `replace` raises at once on a misspelt setting. That is why every adapter setting has a default. A `functools.partial` would not be an adapter until it is called, and ty doesn't check calls to one.
- **`None` keeps the preset's default.** `mass_matrix_adapter=False` turns mass-matrix adaptation off, which is what the deprecated `init="advi"` and `init="map"` translate to.

**Adapters on a block.** In a compound step, each adapter is also put on one block (see [Compound steps](#compound-steps)):

```python
@dataclass(frozen=True)
class OnBlock:
    block: Step                                # one block of a compound step
    adapter: Adapter

    def __post_init__(self):
        check_binding(self.adapter, type(self.block))   # Validation rules 1–2, where the binding is written
```

- **The adapter stays unchanged**, so one preset can go on several blocks.
- **Only a runtime check can verify this link.** An adapter's references are typed `ParamRef[Any, ...]`, so they don't record which step class they belong to.

**`propose` and `final`.** While its phase runs, an adapter's `propose` gives the value the step uses in the next iteration. When the phase ends, `final` gives the value the hyperparameter keeps. The two often differ:

- `DualAveraging` explores with a noisy step size (`propose`) and hands over the average of its iterates (`final`), as today's NUTS does at the end of tuning.
- `WelfordDiag` keeps proposing the covariance it started with while it accumulates draws, and hands over its new estimate at the end. So the covariance changes only between phases. That starting covariance can have any structure, so `WelfordDiag` tunes a `Covariance` field, and only its `final` is sure to return a `DiagCov`.
- `WindowedDiag` proposes a new covariance after every iteration, and its `final` is the latest one.

In the table, "Produces" is what `final` returns. Every covariance adapter tunes a `Covariance` field.

| Adapter | Produces | Inputs | Typical binding |
|---|---|---|---|
| `DualAveraging(target_accept)` | `float` | `accept` | NUTS `step_size`, from `accept_stat` |
| `WindowedDiag()`, `WelfordDiag()` | `DiagCov` | `draws` | NUTS `inverse_mass_matrix`, from `position` |
| `WindowedDiagGrad()`, `FisherDiag()` (nutpie-style) | `DiagCov` | `draws`, `grads` | NUTS `inverse_mass_matrix`, from `position` and `grad` |
| `WindowedFull()`, `WelfordDense()` | `DenseCov` | `draws` | NUTS `inverse_mass_matrix`, from `position` |
| `ShrunkDense(toward="current", shrinkage)` | `DenseCov` | `draws` | NUTS `inverse_mass_matrix`, from `position` |
| `LowRank(k)` | `LowRankPlusDiag` | `draws`, `grads` | NUTS `inverse_mass_matrix`, from `position` and `grad` |
| `AcceptanceScale(target_rate)` | `np.ndarray` | `accept` | Metropolis `scale`, from `accept_prob` |

**`WindowedDiag` and `WelfordDiag` estimate the same thing, but are used differently:**

- `WindowedDiag` is today's PyMC scheme, and covers a whole tuning phase on its own. It keeps two overlapping running estimates. After every iteration it updates the covariance from the older one, and at the end of each of its internal windows it drops the older estimate and starts a new one.
- `WelfordDiag` is Stan's estimator for a single window. It is meant for a sequence of window phases, one adapter per phase, as in `stan_windows` below.
- `WindowedFull` and `WelfordDense` are the dense versions of the two.

### Phases and schedules

```python
@dataclass(frozen=True)
class Phase:
    name: str
    adapters: tuple[Adapter | OnBlock, ...] = ()   # empty → hyperparameters fixed; OnBlock in a compound
    until: StopCondition = Iterations(0)
    warmup: bool = True                         # draws flagged as warmup in storage
    record: bool = True                         # store this phase's draws at all
    allow_invalid_adaptation: bool = False      # explicit opt-out; recorded in the output

Schedule = Sequence[Phase]                      # or an iterable, for schedules built lazily
```

**The default schedule** is the same for every step: a tuning phase with the step's default adapters, then a sampling phase.

```python
def default_schedule(step, tune: int | None, draws: int) -> Schedule:
    tune = step.default_tune if tune is None else tune
    return [Phase("tune", step.default_adapters(), Iterations(tune)),
            Phase("sample", (), Iterations(draws), warmup=False)]
```

**How a phase starts.** Each adapter starts from `current`, the value its hyperparameter has when the phase starts: the value from `default_params` in the first phase, and the value the previous phase handed over after that.

- The initialization result (`hint`) goes only to the adapters of the first phase that tunes their hyperparameter. In later phases they get `hint=None`, and continue from what was handed over.
- So in Stan's windows, the second window's estimator starts from the first window's estimate, and not again from ADVI's approximation.
- The same rule gives Stan's step-size restart at every window, and the hand-off from one estimator to the next.

```python
# Stan's windowed adaptation, for any step: the caller passes the two bound adapters
def stan_windows(
    step_size: Adapter,
    covariance: Adapter,
    tune=1000,
    init_buffer=75,
    term_buffer=50,
    base_window=25,
):
    phases = [Phase("fast-init", (step_size,), Iterations(init_buffer))]
    for i, w in enumerate(doubling_windows(tune - init_buffer - term_buffer, base_window)):
        phases.append(Phase(f"slow-{i}", (step_size, covariance), Iterations(w)))
    phases.append(Phase("fast-term", (step_size,), Iterations(term_buffer)))
    return phases

step_size = DualAveraging(HMCParams.step_size, accept=HMCSignals.accept_stat, target_accept=0.8)
imm, pos = HMCParams.inverse_mass_matrix, HMCSignals.position

schedule = stan_windows(step_size, WelfordDiag(imm, draws=pos))

# Diagonal → strongly regularized dense → full dense
schedule = [
    Phase(
        "diag",
        (step_size, WelfordDiag(imm, draws=pos)),
        Iterations(200),
    ),
    Phase(
        "reg",
        (step_size, ShrunkDense(imm, draws=pos, toward="current", shrinkage=0.9)),
        Iterations(300),
    ),
    Phase(
        "full", (step_size, WelfordDense(imm, draws=pos)), Iterations(500)
    ),
    Phase(
        "sample", (), Iterations(1000) | EssAtLeast(400), warmup=False
    ),
]
```

`ShrunkDense(toward="current")` shrinks toward whatever the previous phase handed over; here that is the diagonal estimate.

### Validation

Binding checks run in two places:

- **`OnBlock` runs rules 1–2 when it is constructed**, against its block's type. A wrong binding on a block fails at the line that wrote it.
- **`check_schedule(schedule, step, init, capabilities)` runs at setup**, before compilation, and checks the whole schedule. It repeats a type checker's binding checks, for users who don't run one and for references built with `ref_to`. It also makes the checks that need the step being sampled, the initialization or the sampler, which can't run earlier. A schedule built lazily can only be checked phase by phase, as it is produced.

Each adapter's step is the block its `OnBlock` names, or the step being sampled for an adapter on its own. `check_schedule` checks that:

1. every adapter's `param` refers to a field of its step's `Params`, or of a class it inherits from, whose type is the adapter's `ValueT`. The field's type is read from its annotation (`ParamField[Covariance]` gives `Covariance`);
2. every `SignalRef` among an adapter's fields refers to a field of its step's `Signals` with the type the adapter's annotation declares (`accept: SignalRef[Any, float]` needs a `float` field);
3. every block an `OnBlock` names is a block of the step being sampled, and in a compound's schedule every adapter is on a block;
4. each phase has at most one adapter per hyperparameter, identified by its block and the field's name;
5. every adapter in a phase with `warmup=False` declares `validity != "warmup_only"`, unless the phase sets `allow_invalid_adaptation=True`;
6. run-scoped stop conditions are only used with samplers that can evaluate them;
7. the sampler accepts a custom schedule at all (`capabilities.custom_schedule`). For example, an external sampler that runs its own warmup can't take a user-defined schedule;
8. every adapter with `needs_approximation=True` is in the first phase that tunes its hyperparameter, and initialization provides an approximation (`provides_approximation`).

Before it, setup rejects an explicit schedule passed together with a step or block that sets an adaptation setting.

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

This defines the reference semantics. PyMC's samplers implement this loop. An external sampler that runs its own loop must produce the same observable behaviour: the same events, in the same order, and output in the same format.

```python
def run_chain(step, bundle, state, schedule, comm, sink, hint, checkpoint=None):
    # Setup turned each phase entry into (b, adapter): b is the position of the adapter's block
    # in a compound step, or None for a single step. A hyperparameter is a (b, field name) pair.
    state = step.init(state, bundle)
    tuned = set()                                           # hyperparameters an earlier phase tuned
    for phase_idx, phase in enumerate(schedule):
        first = {(b, a.param.name) for b, a in phase.adapters} - tuned   # first tuned here: get the hint
        tuned |= first
        if checkpoint and phase_idx < checkpoint.phase_idx:
            continue                                        # resume: skip finished phases
        adapters = [(b, a, a.init(a.param.get(at(state, b).params),
                                  hint if (b, a.param.name) in first else None))
                    for b, a in phase.adapters]
        progress = PhaseProgress.start(phase, checkpoint)
        while not (phase.until.done(progress, sink.run_view) or step.is_finished(state)):
            state, stats, signals = step.transition(state, bundle)
            adapters = [(b, a, a.update(s, a.read(at(signals, b)), comm)) for b, a, s in adapters]
            state = with_params(state, {(b, a.param.name): a.propose(s) for b, a, s in adapters})
            progress = progress.advance()
            sink.record(phase, state, stats)      # → DrawBatch chunks → storage; progress → monitor
            yield state, adapters, progress       # control point: pause, stop, checkpoint
        state = with_params(state, {(b, a.param.name): a.final(s) for b, a, s in adapters})
        sink.phase_end(phase, state)
    sink.final(step.finalize(state, bundle))


def at(x, b):                       # block b's entry of a compound's state or signals
    return x if b is None else x[b]

def with_params(state, values):     # values: {(b, field name): value}
    ...                             # replaces those hyperparameters, block by block
```

The `yield` is the only place where a run can be paused, stopped or checkpointed. Everything between two control points is one synchronous unit of work.

### The sampler contract

```python
class Sampler(ABC):
    name: ClassVar[str]
    capabilities: ClassVar[SamplerCapabilities]

    def __init__(self, **config): ...                      # the loop's settings only; typed, immutable
    def default_step(self, model: Model) -> Step: ...
    def default_chains(self) -> int: ...
    # Raises with a clear reason for a step it can't run, or for a shared setting it can't
    # honour (chains, cores, backend, compile options). Nothing is silently ignored.
    def check_supports(self, step: Step, settings: SharedSettings) -> None: ...
    def start(self, spec: RunSpec, storage: Storage, monitor: Monitor) -> RunHandle: ...


@dataclass(frozen=True)
class RunSpec:                                 # picklable: everything a run needs
    step: Step
    bundle: CompiledBundle
    states: Mapping[int, StepState]
    schedule: Schedule
    hint: InitResult                           # positions and approximation from initialization
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
    def result(self, timeout: float | None = None) -> DataTree: ...   # storage.result(); blocks; re-raises worker errors
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

- A **group** is the set of members of one population run: one SMC run, or one set of chains that share a pooled mass matrix.
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
- Internal samplers emit one batch every N draws. An external sampler that only reports at the end emits one large batch per chain.

### Storage

Every storage subclasses one base class, so PyMC's storages and third-party ones meet the same contract, and the conformance kit checks it. The details are finalized in M3.

```python
class Storage(ABC):
    capabilities: ClassVar[StorageCapabilities]

    @abstractmethod
    def begin(self, spec: RunSpec) -> None: ...
    @abstractmethod
    def write(self, batch: DrawBatch) -> None: ...                   # already postprocessed
    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...   # required of every storage
    @abstractmethod
    def load_checkpoint(self) -> Checkpoint | None: ...              # None for a new run
    @abstractmethod
    def truncate(self, checkpoint: Checkpoint) -> None: ...          # drop draws stored after it, before resuming
    @abstractmethod
    def to_datatree(self) -> DataTree: ...                           # includes the latest checkpoint
    def finalize(self, report: RunReport) -> None: ...               # flush, close, attach the run report
    def result(self) -> DataTree:                                    # what pymc.sample returns
        return self.to_datatree()


@dataclass(frozen=True)
class StorageCapabilities:
    durable: bool              # outlives the process: needed to resume after a crash
    variable_shapes: bool      # can store draws whose shapes change during a run
```

| Storage | Replaces | Durable | Checkpoints |
|---|---|---|---|
| `DataTreeStorage` (default) | today's in-memory `NDArray` backend and `idata_kwargs` | no | the latest one, in a group of the returned `DataTree`; `DataTreeStorage.from_datatree(idata)` resumes from it |
| `ZarrStorage(path)` | `trace=ZarrTrace(...)` | yes | written every `checkpoint_every` draws, and whenever the run stops or `RunHandle.checkpoint()` is called |

- **The checkpoint interval is a storage setting**, because the cost of a checkpoint depends on where it is written. The sampler reads it and checkpoints at the next control point.
- **What a storage can't store raises.** For example, draws whose shapes change during a run go to a storage with `variable_shapes=False`: the write fails with a clear error instead of truncating or padding the data.
- Whether `pymc.sample` returns `result()`, a `DataTree`, or the storage object itself is an open question.

### Monitors

```python
class Monitor(ABC):                  # every method is optional; the defaults do nothing
    def on_start(self, spec: RunSpec, capabilities: SamplerCapabilities) -> None: ...
    def on_progress(self, event: ProgressEvent) -> None: ...          # chain, iteration, phase, monitored stats
    def on_phase_end(self, chain: int, phase: Phase) -> None: ...
    def on_warning(self, warning: SamplingWarning) -> None: ...
    def on_finish(self, status: str, report: RunReport) -> None: ...  # includes convergence checks
```

| Monitor | Purpose |
|---|---|
| `ProgressBarMonitor` | today's progress bar; the default in interactive sessions |
| `FileMonitor(path)` | appends progress, phase ends, warnings and the final report to a log file; the default in non-interactive sessions |
| `CallbackMonitor(fn)` | keeps `callback=` working during its deprecation |
| `SilentMonitor` | what `progressbar=False` selects |

Passing several monitors, `monitor=[ProgressBarMonitor(), FileMonitor("run.log")]`, sends every event to each of them.

**Non-interactive sessions.** A batch job (SLURM, `nohup`, CI) has no terminal to draw a progress bar on, so today its progress is effectively lost. The default monitor is chosen at setup:

- a `ProgressBarMonitor` when stdout is a terminal, or when the code runs in a Jupyter or IPython kernel;
- a `FileMonitor` otherwise. Its file goes next to durable storage if there is one (`run.zarr` → `run.zarr.log`), or into the working directory, named after the run. The path is logged through the `pymc` logger when the run starts.

**How `FileMonitor` writes.**

- **Writes are serialized by a lock.** Events can arrive from the sampler's controller thread, from the caller's thread through the `RunHandle`, and from several runs sharing one log file. The monitor holds a `threading.Lock` around each write, plus an advisory file lock when several processes append to the same file (`fcntl.flock`, or the `filelock` package for portability).
- **Each event is one complete line**, written in a single call and flushed immediately. Lines from different writers never interleave, and the file is up to date if the job is killed.
- **Progress lines are rate-limited** to at most one every `interval` seconds per chain. Phase ends, warnings and the final report are always written.

External samplers will be the stress test for both contracts. They report at a coarser granularity, they can't honour every control, and they bring their own output formats.

## Plugins: external steps and samplers

### What an external library provides

An external library implements the same two contracts as PyMC does, and keeps them separate:

| | Implements | Holds | Runs in | Examples |
|---|---|---|---|---|
| **Step** | `Step`, plus its `State` and `Bundle` | the algorithm's settings | PyMC's samplers, or only the library's own sampler | blackjax kernels; `NutpieNUTS` |
| **Sampler** | `Sampler` | the loop's settings: workers, threads, devices | the library's own runtime | `NutpieSampler`; a `BlackjaxSampler` looping in JAX; numpyro's `MCMC` |

- **blackjax fits both halves naturally.** Its kernels are already pure `(state, key) -> state` functions. Each one wraps as a `Step` that PyMC's samplers can run, and that composes with PyMC's steps in a compound. A `BlackjaxSampler` can also run the same steps in a compiled JAX loop (`lax.scan` over draws, `vmap` over chains).
- **nutpie runs its whole loop in Rust**, so `NutpieNUTS` declares `Requirements(sampler="nutpie")`. Only `NutpieSampler` can run it, and PyMC's samplers reject it in `check_supports` with a clear message. If nutpie later exposes a single transition to Python, the same step can run in PyMC's samplers too, with no change for users.
- **Settings follow the library.** Each step and sampler takes typed constructor arguments, named as its library names them. Only the settings that every step and sampler shares (`draws`, `chains`, `random_seed`, …) have one PyMC name. This is the pattern of the inference-configuration sketch credited under [Initialization](#initialization). A setting the configuration can't use raises.
- **There is one entry point, `pymc.sample`.** A library provides a step and a sampler, not a configured object with its own `sample` method, so the same user code works with every library.

Here is an illustrative sketch. The library calls, settings and capability values are placeholders, not nutpie's actual API.

```python
class NutpieNUTS(Step[...]):
    Bundle = NutpieBundle                    # wraps nutpie's own compiled model
    requirements = Requirements(space="unconstrained", gradient=True,
                                values="continuous", sampler="nutpie")

    def __init__(self, vars=None, *, target_accept=0.8, low_rank_mass_matrix=False, ...): ...
    def default_adapters(self): return ()            # nutpie runs its own warmup
    def default_sampler(self): return NutpieSampler()


class NutpieSampler(Sampler):
    name = "nutpie"
    capabilities = SamplerCapabilities(
        controls=frozenset({"pause", "resume", "cancel"}), progress="per_chunk",
        streaming_output=True, custom_schedule=False, live_stats=True, runs_pymc_steps=False)

    def __init__(self, *, cores=None, ...): ...       # loop settings only

    def check_supports(self, step, settings):
        if not isinstance(step, NutpieNUTS):
            raise UnsupportedStep(...)
        if settings.backend not in (None, "numba", "jax"):
            raise UnsupportedSetting("backend", settings.backend, sampler=self.name)

    def start(self, spec, storage, monitor):
        tune, draws = tune_and_draws(spec.schedule)            # rejects anything but (warmup, sample)
        run = start_library_run(spec.bundle.compiled_model, spec.step, tune=tune, draws=draws,
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
| Adapters | Pure; frozen in non-adaptive phases; a covariance adapter recovers a known covariance on a Gaussian. For `validity="diminishing"`, the parameter changes shrink as declared. |
| Population (SPMD) | Runs with 1, 2 and 3 shards agree; replicated state is identical on every shard. |
| Composition | Invariance still holds inside a compound step alongside a reference step. |
| Storage | Draws and checkpoints round-trip; after a simulated crash, `truncate` plus resume reproduces an uninterrupted run; `to_datatree()` includes the latest checkpoint; what `capabilities` excludes, such as varying shapes, raises. |
| External samplers | The controls listed in `capabilities` work (pause stops progress; cancel leaves valid partial output); events arrive in order; translated batches validate against `DrawBatch`; draw counts match. |
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
- The Storage and Monitor contracts, to be settled in M3, using the external samplers as the stress test.
- What `pymc.sample` returns: a `DataTree`, as today, or the run's `Storage` object, which converts to one. To be decided with the storage prototype in M3.
- Whether `cores` and `progressbar` stay as permanent shorthands, and how long the deprecation period for the other superseded arguments lasts.
- Where `FileMonitor` writes by default when there is no durable storage, and whether its lines are plain text or JSON.
- The public/internal split of modules.
- How a population step that ends through `is_finished`, such as SMC, fits the default schedule of a tuning phase followed by a sampling phase: what `draws` means for it, and which of its iterations are recorded.
- Initialization of a compound step: whether a stage such as Pathfinder may cover only the continuous blocks, holding the others at the root's values, or must raise. Today's ADVI and MAP inits only run when NUTS samples every variable.
- Extending a finished run (more draws) as opposed to resuming a stopped one.
- Bitwise reproducibility across shard counts (allgather-then-reduce versus allreduce of partial results).
- Mixing chain and population steps in one compound (shelved).

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
