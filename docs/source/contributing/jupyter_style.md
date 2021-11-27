(jupyter_style)=
# Jupyter Style Guide

These guidelines should be followed by all notebooks in the documentation, independently of
the repository where the notebook is in (pymc or pymc-examples).

## General guidelines

* Don't use abbreviations or acronyms whenever you can use complete words. For example, write "random variables" instead of "RVs".

* Explain the reasoning behind each step.

* Use the glossary whenever possible. If you use a term that is defined in the Glossary, link to it the first time that term appears in a significant manner. Use [this syntax](https://jupyterbook.org/content/content-blocks.html?highlight=glossary#glossaries) to add a term reference. [Link to glossary source](https://github.com/pymc-devs/pymc/blob/main/docs/source/glossary.md) where new terms should be added.

* Attribute quoted text or code, and link to relevant references.

* Keep notebooks short: 20/30 cells for content aimed at beginners or intermediate users, longer notebooks are fine at the advanced level.

### Variable names

* Above all, stay consistent with variable names within the notebook. Notebooks using multiple names for the same variable will not be merged.

* Use meaningful variable names wherever possible. Our users come from different backgrounds and not everyone is familiar with the same naming conventions.

* Sometimes it makes sense to use Greek letters to refer to variables, for example when writing equations, as this makes them easier to read. In that case, use LaTeX to insert the Greek letter like this `$\theta$` instead of using Unicode like `θ`.

* If you need to use Greek letter variable names inside the code, please spell them out instead of using unicode. For example, `theta`, not `θ`.

* When using non meaningful names such as single letters, add bullet points with a 1-2 sentence description of each variable below the equation where they are first introduced.


## First cell
The first cell of all example notebooks should have a MyST target, a level 1 markdown title (that is a title with a single `#`) followed by the post directive.
The syntax is as follows:

```markdown
(notebook_id)=
# Notebook Title

:::{post} Aug 31, 2021
:tags: tag1, tag2, tags can have spaces, tag4
:category: level
:author: Alice Abat, Bob Barceló
:::
```

The date should correspond to the latest update/execution date, at least roughly (it's not a problem if the date is a few days off due to the review process before merging the PR). This will allow users to see which notebooks have been updated lately and will help the PyMC team make sure no notebook is left outdated for too long.

The [MyST target](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html#targets-and-cross-referencing)
is important to ease referencing and linking notebooks between each other.

Tags can be anything, but we ask you to try to use [existing tags](https://github.com/pymc-devs/pymc/wiki/Categories-and-Tags-for-PyMC-Examples)
to avoid the tag list from getting too long.

Each notebook should have a single category indicating the level of the notebook.
Choose a category from [existing categories](https://github.com/pymc-devs/pymc/wiki/Categories-and-Tags-for-PyMC-Examples#categories).

Authors should list people who authored, adapted or updated the notebook. See {ref}`jupyter_authors`
for more details.

## Code preamble

In a cell just below the cell where you imported matplotlib and/or ArviZ (usually the first one),
set the ArviZ style to darkgrid (this has to be in another cell than the matplotlib import because of the way matplotlib sets its defaults):

```python
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")
```

A good practice _when generating synthetic data_ is also to set a random seed as above, to improve reproducibility. Also, please check convergence (e.g. `assert all(r_hat < 1.03)`) because we sometime re-run notebooks automatically without carefully checking each one.

## Reading from file

Use a `try... except` clause to load the data and use `pm.get_data` in the except path. This will ensure that users who have cloned pymc-examples repo will read their local copy of the data while also downloading the data from github for those who don't have a local copy. Here is one example:

```python
try:
    df_all = pd.read_csv(os.path.join("..", "data", "file.csv"), ...)
except FileNotFoundError:
    df_all = pd.read_csv(pm.get_data("file.csv"), ...)
```

## pre-commit and code formatting
We run some code-quality checks on our notebooks during Continuous Integration. The easiest way to make sure your notebook(s) pass the CI checks is using [pre-commit](https://github.com/pre-commit/pre-commit). You can install it with

```bash
pip install -U pre-commit
```

and then enable it with

```bash
pre-commit install
```

Then, the code-quality checks will run automatically whenever you commit any changes. To run the code-quality checks manually, you can do, e.g.:

```bash
pre-commit run --files notebook1.ipynb notebook2.ipynb
```

replacing `notebook1.ipynb` and `notebook2.ipynb` with any notebook you've modified.

NB: sometimes, [Black will be frustrating](https://stackoverflow.com/questions/58584413/black-formatter-ignore-specific-multi-line-code/58584557#58584557) (well, who isn't?). In these cases, you can disable its magic for specific lines of code: just write `#fmt: on/off` to disable/re-enable it, like this:

```python
# fmt: off
np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)
# fmt: on
```

(jupyter_authors)=
## Authorship and attribution
After the notebook content finishes, there should be an `## Authors` section with bullet points
to provide attribution to the people who contributed to the the general pattern should be:

```markdown
* <verb> by <author> on <date> ([repo#PR](https://link-to.pr))
```

where `<verb>` must be one listed below, `<author>` should be the name (multiple people allowed)
which can be formatted as hyperlink to personal site or GitHub profile of the person,
and `<date>` should preferably be month and year.

authored
: for notebooks created specifically for pymc-examples

adapted
: for notebooks adapted from other sources such as books or blogposts.
  It will therefore follow a different structure than the example above
  in order to include a link or reference to the original source:

  ```markdown
  Adapted from Alice's [blogpost](blog.alice.com) by Bob and Carol on ...
  ```

re-executed
: for notebooks re-executed with a newer PyMC version without significant changes to the code.
  It can also mention the PyMC version used to run the notebook.

updated
: for notebooks that have not only been re-executed but have also had significant updates to
  their content (either code, explanations or both).

some examples:

```markdown
* Authored by Chris Fonnesbeck in May, 2017 ([pymc#2124](https://github.com/pymc-devs/pymc/pull/2124))
* Updated by Colin Carroll in June, 2018 ([pymc#3049](https://github.com/pymc-devs/pymc/pull/3049))
* Updated by Alex Andorra in January, 2020 ([pymc#3765](https://github.com/pymc-devs/pymc/pull/3765))
* Updated by Oriol Abril in June, 2020 ([pymc#3963](https://github.com/pymc-devs/pymc/pull/3963))
* Updated by Farhan Reynaldo in November 2021 ([pymc-examples#246](https://github.com/pymc-devs/pymc-examples/pull/246))
```

and

```markdown
* Adapted from chapter 5 of Bayesian Data Analysis 3rd Edition {cite:p}`gelman2013bayesian`
  by Demetri Pananos and Junpeng Lao on June, 2018 ([pymc#3054](https://github.com/pymc-devs/pymc/pull/3054))
* Reexecuted by Ravin Kumar with PyMC 3.6 on March, 2019 ([pymc#3397](https://github.com/pymc-devs/pymc/pull/3397))
* Reexecuted by Alex Andorra and Michael Osthege with PyMC 3.9 on June, 2020 ([pymc#3955](https://github.com/pymc-devs/pymc/pull/3955))
* Updated by Raúl Maldonado 2021 ([pymc-examples#24](https://github.com/pymc-devs/pymc-examples/pull/24), [pymc-examples#45](https://github.com/pymc-devs/pymc-examples/pull/45) and [pymc-examples#147](https://github.com/pymc-devs/pymc-examples/pull/147))
```

## References
References should be added to the [`references.bib`](https://github.com/pymc-devs/pymc-examples/blob/main/examples/references.bib) file in bibtex format, and cited with [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/) within the notebook text wherever they are relevant.

The references in the `.bib` file should have as id something along the lines `authorlastnameYEARkeyword` or `libraryYEARkeyword` for documentation pages, and they should be alphabetically sorted by this id in order to ease finding references within the file and preventing adding duplicate ones.

References can be cited twice within a single notebook. Two common reference formats are:

```
{cite:p}`bibtex_id`  # shows the reference author and year between parenthesis
{cite:t}`bibtex_id`  # textual cite, shows author and year without parenthesis
```

which can be added inline, within the text itself. At the end of the notebook, add the bibliography with the following markdown

```
## References

:::{bibliography}
:filter: docname in docnames
:::
```

or alternatively, if you wanted to add extra references that have not been cited within the text, use:

```
## References

:::{bibliography}
:filter: docname in docnames

extra_bibtex_id_1
extra_bibtex_id_2
:::
```

## Watermark
Once you're finished with your NB, add a very last cell with [the watermark package](https://github.com/rasbt/watermark). This will automatically print the versions of Python and the packages you used to run the NB -- reproducibility rocks! Here is some example code. Note that the `-p` argument may not be necessary (or it may need to have different libraries as input), but all the other arguments must be present.

```python
%load_ext watermark
%watermark -n -u -v -iv -w -p theano,xarray
```

This second to last code cell should be preceded by a markdown cell with the `## Watermark` title only so it appears in the table of contents.

`watermark` should be in your virtual environment if you installed our `requirements-dev.txt`. Otherwise, just run `pip install watermark`. The `p` flag is optional but should be added if Theano (or Aesara if in `v4`) or xarray are not imported explicitly.
This will also be checked by `pre-commit` (because we all forget to do things sometimes 😳).

## Epilogue
The last cell in the notebooks should be a markdown cell with exactly the following content:

```
:::{include} ../page_footer.md
:::
```

The only exception being notebooks that are not on the usual place and therefore need to
update the path to page footer for the include to work.

---

You're all set now 🎉 You can push your changes, open a pull request, and, once it's merged, rest with the feeling of a job well done 👏
Thanks a lot for your contribution to open-source, we really appreciate it!
