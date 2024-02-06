# Contributing

PyMC is an open source, collective effort.
There are many ways in which you can help make it better.
And all of them are welcome!

## Contribute as an individual

PyMC is a joint effort of many people, each contributing to the areas they like
and have some expertise in, coordinating to try and cover all tasks.

Coding and documentation are the most common types of contributions, but
there are many more things that you can do to help PyMC which are just as
important. Moreover, both code and docs require submitting PRs via GitHub
to some of the repositories under the [pymc-devs](https://github.com/pymc-devs) organization, and
while we have a {ref}`pr_tutorial` guide available, GitHub might not be
everyone's cup of tea. If that is your case, don't worry, you will be
more than welcome if you want to help.

:::{tip}
Contact us on [Discourse](https://discourse.pymc.io/) if you want to contribute to the project but are not sure where you can contribute or how to start.

We also host office hours regularly to provide more support, especially to contributors.
If you are interested in participating [subscribe to the `office-hours` tag](https://discourse.pymc.io/tag/office-hours) on Discourse.
:::

Below there are some examples of non code nor doc contributions that could serve as an inspiration.
If you have other ideas let us know on [Discourse](https://discourse.pymc.io/) to see if we can make it happen too.

* Report a bug or make a suggestion for improvement by [opening an issue in Github](https://github.com/pymc-devs/pymc/issues/new/choose)
* Answer questions on [Discourse](https://discourse.pymc.io/)
* Teach about PyMC and advertise best practices by writing blogs or giving talks
* Help plan PyMCon
* Help with outreach and marketing. This could include for example reaching out to potential sponsor
  companies, to people who could use PyMC in their work or making sure that academics who use PyMC
  cite it correctly in their work
* Help with our fundraising efforts
* Add timestamps to [videos from PyMCon](https://github.com/pymc-devs/video-timestamps)

### Contribute via Pull Requests on GitHub
We have a {ref}`pr_tutorial` and a {ref}`pr_checklist` page to help in all the steps of the contributing
process, from before your first ever contribution to regular contributions as a core contributor.

(pr_etiquette)=
#### Etiquette for code contributions
* When you start working working on an issue, open a `Draft` pull request as soon as you make your first commit (see {ref}`pr_tutorial`).
* Before opening a PR with a new feature, please make a proposal by opening an [issue](https://github.com/pymc-devs/pymc/issues) or [Discussion](https://github.com/pymc-devs/pymc/discussions) with the maintainers. Depending on the proposal we might direct you to other places such as [`pymc-experimental`](https://github.com/pymc-devs/pymc-experimental) or [`pymc-examples`](https://github.com/pymc-devs/pymc-examples).
* Any issue without an open pull request is available for work.
   * If a pull request has no recent activity it may be closed, or taken over by someone else.
   * The specific timeframe for "recent" is hard to define as it depends on the contributor the specific code change, and other contextual factors. As a rule of thumb in a normal pull request with no other blockers there is typically activity every couple of days.
   * The core devs will make their best judgement when opting to close PRs or reassign them to others.
* If unsure if an issue ticket is available feel free to ask in the issue ticket. Note however, that per the previous point an open pull request is way to claim an issue ticket. Please do not make unrealistic pledges in the issue tickets.
* It's okay if you are delayed or need to take a break, but please leave a comment in the pull request if you cannot get it to a state where it can be merged. Depending on the change (urgent bugfix vs. new feature) the core devs can determine if the PR needs to be reassigned to get the work done.


#### Code related contributions
Join the discussion or submit a solution for an open issue. [See open issues](https://github.com/pymc-devs/pymc/issues)

#### Documentation related contributions

See all open issues in documentation [here](https://github.com/pymc-devs/pymc/issues?q=is%3Aissue+is%3Aopen+label%3A%22docs%22+)

:::{admonition} New to the open source space?
:class: tip

If you are not sure where or how to start, take a look at the [sprint materials](https://pymc-data-umbrella.xyz/en/latest/sprint/docstring_tutorial.html)
(even if you plan on contributing on your own outside of sprint events).
They are the most detailed guide available on contributing to PyMC also
with advice on which contributions are good starting points.
:::

## Contribute as an institution

Institutions can contribute in the following ways:

- By becoming [Institutional Partners](https://github.com/pymc-devs/pymc/blob/main/GOVERNANCE.md#institutional-partners-and-funding)
- By becoming [Sponsors](https://github.com/pymc-devs/pymc/blob/main/GOVERNANCE.md#sponsors)

Contact PyMC at pymc.devs@gmail.com for more information.


:::{toctree}
:hidden:
:maxdepth: 1
:caption: Tutorials

pr_tutorial
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: How-to guides

implementing_distribution
build_docs
docker_container
running_the_test_suite
review_pr_pymc_examples
using_gitpod
:::

:::{toctree}
:hidden:
:maxdepth: 1
:caption: Reference content

python_style
jupyter_style
pr_checklist
release_checklist
:::


:::{toctree}
:hidden:
:maxdepth: 1
:caption: In depth explanations

versioning_schemes_explanation
:::
