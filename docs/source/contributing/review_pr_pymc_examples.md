(review_pr_pymc_examples)=
# Review a PR on pymc-examples

The target audience for this page are reviewers of PRs in pymc-examples
repo. It mostly gathers resources from its
[contributing guide](https://github.com/pymc-devs/pymc-examples/blob/main/CONTRIBUTING.md),
[PR template](https://github.com/pymc-devs/pymc-examples/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
and the {ref}`jupyter_style` page to
centralize all resources for reviewers and put special emphasis on
reviewer tasks and responsibilities.

:::{important}
The most important guideline is the following: **When you aren't _completely_
sure about something communicate it, ask around and (re)read the documentation**.
:::

[pymc-examples](https://github.com/pymc-devs/pymc-examples/) is a huge collection of notebooks, covering many fields
and applications of PyMC and Bayesian statistics. In addition,
its HTML generation infrastructure has been improved by specialized
people who were also paid so they could dedicate themselves to this task.
It is perfectly fine to not know about the topic, not know about
the tech stack being used or to not be sure about either.
You can still review, either focus on your expertise or take some
time going through all the available resources to learn about the parts
you are not sure about.

(review_pr/scope)=
## 1. Define the scope
Before starting you need to make sure you understand the scope of the PR and
to set the scope of your own review.
If the scope isn't clear from the PR description, ask the author to update it!

There are many valid contributions that can be done to the notebooks in pymc-examples.
They can be focused on code, styling, wording...
pymc-examples needs to have regular updates and in many cases it doesn't matter if
these are partial and don't update everything that can be updated in the notebook.

Not all partial updates are valid nor helpful though. If doing updates
to the code for example, most of the wording will probably need no changes,
but there may be some sections where the explanation is specifically about the code
used. We don't know when the next partial update will come, so PRs should merge
working and coherent notebooks. If the code is updated to use `Potential` instead
of `DensityDist` and the explanation mentions `DensityDist`, that specific sentence
needs to be updated too.

A PR that aims to update everything about a notebook
could easily have 3 or more reviewers, each covering different aspects of the
example like Aesara usage, writing and explanation of the concepts used,
ArviZ usage, styling with MyST+Sphinx, or structuring and scope of the notebook.

Unless you plan to review everything, start your review mentioning what your
review covers (or skips).

Part of that is also making sure that the PR description links to the relevant issue(s).

(review_pr/initial_pass)=
## 2. Try to be concise and clear
Try to be concise and to the point with your reviews.
**Make sure to leave actionable responses.**
You'll need less time to review and the PR author will need less time to go over the review.
Here are some examples about what this does:

* Skim the notebook from top to bottom before starting your review.
  This will prevent you from spending some time writing about missing content
  and suggesting how to add it only to find out it is not missing but badly structured.
* Do not proofread the text until you are sure no large rewritings will be needed.
  What is the point of reviewing/fixing a typo when the whole paragraph is to be rewritten?
* Include some rationale about your comments. Contributors come from a wide range
  of places, fields and personal contexts and English might not be their (nor your) first language.
  Adding the reasons behind a suggestion goes a long way in making sure the suggestion
  is clear and makes it more probable that the author doesn't repeat this pattern again.

and doesn't mean:

* Avoiding mentioning good things from the PR or not thanking the PR author.
* Being cryptic or writing badly

## 3. Review the code and supporting text
* Check the intended level of the notebook, both when it comes to code and to text.
  i.e. beginner notebooks should sacrifice performance for the sake of clarity without even needing
  an explanation, intermediate and advanced ones should not.
* In all cases, however, remember that this code is written to be read!
  There is little to gain from obscure one-liners and much to lose, even in
  intermediate or advanced notebooks. If some code is necessary but not
  very relevant and you think takes up too much space, hide that cell/input/output
  under a toggle button.
* Make sure that the text is relevant, it explains the code blocks when needed and
  is up to date with the code.
* Check diffs and output on ReviewNB

## 4. Review the styling and formatting
* Ignore ReviewNB. ReviewNB renders notebooks incorrectly when there are diffs involving formatting,
  it can add non-existent HTML tags to the rendered view, mess up links... Moreover, we don't
  want to render the notebook by itself but as a page on the example gallery.
* Check the readthedocs preview and use the MyST notebook format to see the raw text diff
  and comment on it when it comes to formatting.
* In general check that the style described in {ref}`jupyter_style` is followed.

For the time being (while we rerun the notebooks with v4 and update the docs to new formatting)
please ensure all of the following:

* There are **NO** URLs pointing to PyMC/ArviZ/Aesara docs
* There is a post directive and MyST target at the top of the notebook.
* The notebook is being checked by pre-commit (it should not appear in any exclude section in `.pre-commit-config.yaml`)
* No watermark (this is already CI breaking but is still included here for full context)

They are very specific and should be part of the PR scope even if the PR author
didn't intend to originally. Think of this as CI breaking.

## 5. Check CI
* CI in pymc-examples is very specific and has never had a false failure yet.
* What it does however is skipping some files from given checks! Make sure to check the
  file is not excluded from CI when reviewing.

## Checklist
This might be moved to a comment added by a bot to every new PR, see [pymc-examples#288](https://github.com/pymc-devs/pymc-examples/issues/288)

* Check PR description for clarity and links to relevant issues
* Define scope of review
* Leave actionable comments with rationale
* Check code, outputs and supporting text in ReviewNB
* Check styling and rendering in readthedocs preview and MyST notebook representation
* Check there are:
  - **No** URLs pointing to PyMC/Aesara/ArviZ docs
  - A post directive with tags and categories and MyST target at the top of the notebook
  - A watermark with all relevant libraries for reproducibility at the bottom of the notebook
* Check CI is passing and the notebook is being checked by pre-commit
* Proofread changes (once no significant rewriting is needed)
