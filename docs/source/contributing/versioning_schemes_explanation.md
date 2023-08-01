# PyMC websites versioning

The PyMC project has 3 main websites related directly to the
PyMC library. These exist in addition to more specific websites like PyMCon, sprints or the docs of pymc-experimental.
This guide explains their relation and what type of content should go on each of the websites.

:::{attention}
All 3 websites share the same nabvar to give the appearance of
a single website to users, but their generation process is completely independent from one another.
:::

The content that appears under the `Home` and `Community` sections is generated from the source at [pymc.io](https://github.com/pymc-devs/pymc.io);
the content that appears under `Learn`, `API` and `Contributing`
is generated from the source at [pymc](https://github.com/pymc-devs/pymc/tree/main/docs/source) and the content under `Examples` is generated from [pymc-examples](https://github.com/pymc-devs/pymc-examples).

The unversioned website is the main PyMC website, it contains the landing page and it is the parent project on ReadTheDocs.
The other two are configured as subprojects of the unversioned
website so that when we use the search bar, the keyword is searched
across all 3 websites.

## Unversioned website: `www.pymc.io`
We publish the unversioned website at `www.pymc.io`.
It has no language or version indicator.

The unversioned website is the main landing page and hosts content
that is relevant to the PyMC library or project but not
tied to a specific version. For example the PyMC ecosystem,
community resources like the calendar or the code of conduct and
the PyMC blog.

## Versioned docs: `www.pymc.io/projects/docs`
We publish the versioned docs at `www.pymc.io/projects/docs`
plus the language and version indicators. By default `/en/stable/`.

The versioned docs **are synced with PyMC releases** and contain a handful of
guides about core functionality, the API documentation and the
contributing guides.
Being synced with the releases means that the same exact code
we tag on GitHub and that goes to PyPI is used to generate the docs.
Therefore, even if the `latest` version is updated with every commit,
the default website `stable` is only updated when a new release is published.

## Example gallery: `www.pymc.io/projects/examples`
We publish the example gallery at: `www.docs.pymc.io/projects/examples`
plus the language and snapshot indicator, by default `/en/latest/`.

We will update the notebooks in the example gallery regularly
and publish the updates to the example gallery website with each commit.
We recommend using the example gallery as an unversioned living
resource, but we will also provide snapshots every few months for cases such as books
that need links to a stable resource.

You can access the snapshots from the menu at the bottom right of the page.
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
