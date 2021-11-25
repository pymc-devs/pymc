# Build documentation locally

To build the docs, run these commands at pymc repository root:

TODO: fix makefile

```bash
$ pip install -r requirements-dev.txt  # Make sure the dev requirements are installed
$ cd docs/source
$ make html  # Build docs
$ python -m http.server --directory ../_build/html  # Render docs
```

Check the printed url where docs are being served and open it.
