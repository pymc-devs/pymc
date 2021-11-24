# PyMC Release workflow
+ Track all relevant issues and PRs via a **version-specific [milestone](https://github.com/pymc-devs/pymc/milestones)**
+ Make sure that there are no major known bugs that should not be released
+ Make a PR to **bump the version number** in `__init__.py` and edit the `RELEASE-NOTES.md`.
  + :::{important}
    Please don't name it after the release itself, and remember to push to your own fork like an ordinary citizen.
    :::
  + Create a new "vNext" section at the top
  + Edit the header with the release version and date
  + Add a line to credit the release manager like in previous releases
+ After merging the PR, check that the CI pipelines on master are all ✔
+ Create a Release with the Tag as ´v1.2.3´ and a human-readable title like the ones on previous releases

After the last step, the [GitHub Action "release-pipeline"](https://github.com/pymc-devs/pymc/blob/master/.github/workflows/release.yml) triggers and automatically builds and publishes the new version to PyPI.

## Troubleshooting
+ If for some reason, the release must be "unpublished", this is possible by manually deleting it on PyPI and GitHub. HOWEVER, PyPI will not accept another release with the same version number!
+ The `release-pipeline` has a `test-install-job`, which can fail if the PyPI index did not update fast enough.

## Post-release steps
+ Head over to [Zenodo](https://zenodo.org/record/4603970) and copy the version specific DOI-bade into the [release notes](https://github.com/pymc-devs/pymc/releases)
+ Rename and close the release milestone and open a new "vNext" milestone
+ Monitor the update the [conda-forge/pymc-feedstock](https://github.com/conda-forge/pymc-feedstock) repository for new PRs. The bots should automatically pick up the new version and open a PR to update it. Manual intervention may be required though (see the repos PR history for examples).
+ Re-run notebooks with the new release (see https://github.com/pymc-devs/pymc-examples)
+ Make sure the new version appears at the website and that [`docs.pymc.io/en/stable`](https://docs.pymc.io/en/stable) points to it.
