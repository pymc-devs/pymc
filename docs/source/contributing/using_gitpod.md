(using_gitpod)=
# Using Gitpod

## About Gitpod
[Gitpod](https://www.gitpod.io/) is a browser-based development environment.

These are some benefits to using Gitpod:

- Bypass local computer configuration and technical issues
- Save time by using a pre-configured virtual environment for contributing to open source
- Save space on your local computer
- Alleviate delays in situations with low internet bandwidth

## Using Gitpod to Contribute to PyMC

These instructions are for contributing specifically to the [pymc-devs/pymc](https://github.com/pymc-devs/pymc) repo.

### Gitpod Workflow

1. Fork the pymc repo: [https://github.com/pymc-devs/pymc](https://github.com/pymc-devs/pymc)

2. Create a Gitpod account. You can login and authorize access via your GitHub account:  [https://gitpod.io/](https://gitpod.io/)

**NOTE:** Gitpod will show up as an authorized application in your account here:  [https://github.com/settings/applications](https://github.com/settings/applications)

3. Grant GitHub / Gitpod integration permissions.

a) Go to: [https://gitpod.io/user/integrations](https://gitpod.io/user/integrations)

b) Select GitHub and then "edit permissions"

c) Select these permission: user:email, public_repo, repo, workflow

::::{grid-item-card} Gitpod and GitHub Integration
:img-top: gitpod/gitpod_integration.png

::::

4. Within Gitpod, create a "New Workspace".  Here you will want to select the forked pymc repo. If you don't see it, you can paste into the "Context URL" your forked repo path.  For example:  `https://github.com/reshamas/pymc`.  Then select "New Workspace".

**NOTE:** Gitpod will pull a container and set up the workspace.  It will take a few minutes for the container to build.

5. Once Gitpod is up and running, the interface is similar to a Visual Studio Code (VSC) interface, which will appear in your browser. You will observe installation notices in the terminal window.  This can take 5-10 minutes. Once that is complete, the terminal will indicate you are on the "(base)" environment on Gitpod with your forked repo.

Here is an example:

```bash
(base) gitpod@reshamas-pymc-0ygu5rf74md:/workspace/pymc$
```

.. admonition:: This working environment has been set up with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) which is a small, pure-C++ executable with enough functionalities to bootstrap fully functional conda-environments.

6. Check that your git remotes are correct with `git remote -v` at the terminal.

Example:

```bash
(base) gitpod@reshamas-pymc-0ygu5rf74md:/workspace/pymc$ git remote -v
origin  https://github.com/reshamas/pymc.git (fetch)
origin  https://github.com/reshamas/pymc.git (push)
upstream        https://github.com/pymc-devs/pymc.git (fetch)
upstream        https://github.com/pymc-devs/pymc.git (push)
(base) gitpod@reshamas-pymc-0ygu5rf74md:/workspace/pymc$
```

7. Check which version of python and pymc are being used at the terminal.

a) Check version of pymc: `pip list | grep pymc`

Example:

```bash
(base) gitpod@reshamas-pymc-vpfb4pvr90z:/workspace/pymc$ pip list | grep pymc
pymc                          5.1.0       /workspace/pymc
pymc-sphinx-theme             0.1
(base) gitpod@reshamas-pymc-vpfb4pvr90z:/workspace/pymc$
```

b) Check version of python: `python3 --version`

Example:

```bash
(base) gitpod@reshamas-pymc-vpfb4pvr90z:/workspace/pymc$ python3 --version
Python 3.11.0
(base) gitpod@reshamas-pymc-vpfb4pvr90z:/workspace/pymc$
```

### Reminders
At the terminal, before beginning work, remember to:

1. Create a feature branch: `git checkout -b feature-branch`
1. Work on a file
1. Follow the Git workflow
```bash
git add file_name
git commit -m 'message'
git push origin feature-branch
```

### Gitpod Notes
The Gitpod free plan currently allow 500 free credits (50 hours of standard workspace usage) per month. Usage information can be found in the [Gitpod billing section](https://gitpod.io/user/billing).
