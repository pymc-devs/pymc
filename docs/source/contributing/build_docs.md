# Build documentation locally

To build the docs, run these commands at pymc repository root:

```bash
$ pip install -r requirements-dev.txt  # Make sure the dev requirements are installed
$ pip install numpyro  # Make sure `sampling_jax` docs can be built
$ make clean  # clean built docs from previous runs and intermediate outputs
$ make html   # Build docs
$ python -m http.server --directory docs/_build/  # Render docs
```

Check the printed url where docs are being served and open it.

The `make clean` step is not always necessary, if you are working on a specific page
for example, you can rebuild the docs without the clean step and everything should
work fine. If you are restructuring the content or editing toctrees, then you'll need
to execute `make clean`.

A good approach is to skip the `make clean`, which makes
the `make html` blazing fast and see how everything looks.
If something looks strange, run `make clean` and `make html` one after the other
to see if it fixes the issue before checking anything else.
