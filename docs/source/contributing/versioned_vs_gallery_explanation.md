# PyMC example gallery

**Welcome to the PyMC example gallery!**

The PyMC example gallery is a collection of Jupyter notebooks
about PyMC and its usage. These notebooks can be tutorials,
case studies or in-depth explanations.
Some notebooks were originally blog posts, others were adapted from books
and some were written specifically for this example gallery.
This homepage explains the organization of the website and provides
some pointers on how to navigate it.

The collection contains more than 90 notebooks. We are therefore unable
to ensure notebooks are updated and re-executed at the same pace we
publish PyMC releases. Consequently, PyMC has two documentation
websites: the versioned docs and the example gallery

## Versioned docs
We publish the versioned docs directly at `docs.pymc.io`. The only additions
to the url are the language and version of the documentation.

The versioned docs are synced with PyMC releases and contain a handful of
guides about core functionality and the API documentation among other things.

## Example gallery
We publish the example gallery as a sub-project of the versioned docs: `docs.pymc.io/projects/examples`
This means that the building process, language and version of the
example gallery are independent from the versioned docs. However,
the {ref}`search bar in the versioned docs <pymc:search>` searches
both the versioned docs and the example gallery at the same time
(but not the other way around).

We will update the notebooks in the example gallery regularly
and publish the updates to the example gallery website with each commit.
We recommend using the example gallery as an unversioned living
resource, but we will also provide snapshots every few months for cases such as books
that need links to a stable resource.

You can access the snapshots from the version menu at the bottom right of the page.
A version number in the `YYYY.0M.MICRO` format identifies the time the snapshot was published.

---

Notebooks are treated as blog posts. The metadata of each notebook
describes its topics and type of content via tags and categories
and the last update date. We believe that tags and categories
ease and improve navigation (as opposed to a fixed topic division/hierarchy).
In addition, we also provide a list of recent updates and a search bar in the
navigation bar at the top of the page.

:::{caution}
The website is still under construction. Thus, not all notebooks have been updated
to include all the relevant metadata. Those notebooks can only be reached
from the search bar.
:::

### Categories
Notebooks have at most two categories, one indicating the level of the
notebook and another indicating the type of content according to the
[diataxis framework](https://diataxis.fr/). The left sidebar
shows all 7 categories (3 levels + 4 types) at all times. You can click
there to reach the page listing all the notebooks in that category.
If a page has some categories in its metadata they are highlighted in green
in the category list.

### Tags
Notebooks can have any number of tags. Each tag represents a specific topic
of potential interest to readers or a pymc object used within that notebook.

The left sidebar shows all tags at all times. Like categories, they can be clicked
on to reach the page listing all notebooks that contain the tag. If a notebook
has tags in its metadata they are listed on the right sidebar after the {fas}`tags` icon.

:::{toctree}
:maxdepth: 1
:hidden:
gallery
blog
object_index/index
:::
