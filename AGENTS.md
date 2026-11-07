# Contributing to PyMC

Prioritize useful improvements to supported modeling workflows. Correctness,
performance, compatibility, maintenance, and reviewer attention all matter.

## Is this a bug?

- Identify the intended contract and a realistic user impact. Distinguish wrong
  results with valid inputs from undefined behavior, unsupported features, and
  intentional design choices.
- Reproduce on current code through the actual PyMC workflow. Model gradients
  include pre-gradient rewrites; raw symbolic differentiation is not equivalent.
  Account for normal compilation, transformations, and backend behavior.
- Check the implementation, related work, and maintainer discussions. A stale
  branch or an already-fixed problem does not require another patch.
- Respect the division of responsibilities with PyTensor. An unstable-looking
  expression or an artificial execution path alone does not justify duplicating
  compiler stabilization or adding defensive code in PyMC.

## Is fixing it an acceptable trade-off?

- Weigh the practical benefit against runtime, compilation, complexity,
  compatibility, and maintenance costs. Do not burden common valid uses with
  checks or branches for speculative edge cases.
- Assess numerical accuracy and performance across relevant ranges. Improving one
  example can degrade another or interfere with existing optimizations.
- Address shared causes at the appropriate layer. Prefer general solutions and
  coherent policy changes to accumulating local workarounds.
- Treat user-visible behavior changes as design decisions, even when the diff is
  small. Decline or defer changes whose benefits do not justify their costs.
- Verify automated findings independently and keep work bounded. Consolidate
  related findings instead of opening an issue or PR for each speculative failure.
  Maintainer attention is part of a contribution's cost.

Use focused, proportionate tests of the actual failing workflow. Explain the user
impact, evidence, and trade-offs concisely. Distinguish a rejected proposal from
work that needs separate investigation or review.
