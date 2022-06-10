# Guidelines for Contributing

Thank you for being interested in contributing to PyMC. PyMC is an open source, collective effort, and everyone is welcome to contribute. There are many ways in which you can help make it better. Please check the latest information for contributing to the PyMC project on [this guidelines](https://docs.pymc.io/en/latest/contributing/index.html).

Quick links
-----------

* [Pull request (PR) step-by-step ](https://docs.pymc.io/en/latest/contributing/pr_tutorial.html)
* [Pull request (PR) checklist](https://docs.pymc.io/en/latest/contributing/pr_checklist.html)
* [Python style guide with pre-commit](https://docs.pymc.io/en/latest/contributing/python_style.html)
* [Running the test suite](https://docs.pymc.io/en/latest/contributing/running_the_test_suite.html)
* [Submitting a bug report or feature request](https://github.com/pymc-devs/pymc/issues)

<!-- Commented out because our Docker image is outdated/broken.
## Developing in Docker

We have provided a Dockerfile which helps for isolating build problems, and local development.
Install [Docker](https://www.docker.com/) for your operating system, clone this repo, then
run `./scripts/start_container.sh`. This should start a local docker container called `pymc`,
as well as a [`jupyter`](http://jupyter.org/) notebook server running on port 8888. The
notebook should be opened in your browser automatically (you can disable this by passing
`--no-browser`). The repo will be running the code from your local copy of `pymc`,
so it is good for development.

You may also use it to run the test suite, with

```bash
$  docker exec -it pymc  bash # logon to the container
$  cd ~/pymc/tests
$  . ./../../scripts/test.sh # takes a while!
```

This should be quite close to how the tests run on TravisCI.

If the container was started without opening the browser, you
need the notebook instances token to work with the notebook. This token can be
accessed with

```
docker exec -it pymc jupyter notebook list
```
-->
