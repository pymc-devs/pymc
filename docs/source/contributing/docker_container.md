(docker_container)=
# Running PyMC in Docker

We have provided a Dockerfile which helps for isolating build problems, and local development.
Install [Docker](https://www.docker.com/) for your operating system, clone this repo, then
run the following commands to build a `pymc` docker image.

```bash
cd pymc
bash scripts/docker_container.sh build
```

After successfully building the docker image, you can start a local docker container called `pymc` either from `bash` or from [`jupyter`](http://jupyter.org/) notebook server running on port 8888.

```bash
bash scripts/docker_container.sh bash # running the container with bash
bash scripts/docker_container.sh jupyter # running the container with jupyter notebook
```
