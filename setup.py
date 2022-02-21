#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import re

from codecs import open
from datetime import datetime, timezone
from os.path import dirname, join, realpath

from setuptools import find_packages, setup

DESCRIPTION = "Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning with Aesara"
AUTHOR = "PyMC Developers"
AUTHOR_EMAIL = "pymc.devs@gmail.com"
URL = "http://github.com/pymc-devs/pymc"
LICENSE = "Apache License, Version 2.0"
NIGHLTY = "BUILD_PYMC_NIGHTLY" in os.environ

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, "README.rst"), encoding="utf-8") as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

test_reqs = ["pytest", "pytest-cov"]


def get_distname(nightly_build=False):
    distname = "pymc"
    if nightly_build:
        distname = f"{distname}-nightly"

    return distname


def get_version(nightly_build=False):
    version_file = join("pymc", "__init__.py")
    lines = open(version_file).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            version = mo.group(1)

            if nightly_build:
                suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
                version = f"{version}{suffix}"

            return version

    raise RuntimeError(f"Unable to find version in {version_file}.")


if __name__ == "__main__":
    setup(
        name=get_distname(NIGHLTY),
        version=get_version(NIGHLTY),
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/x-rst",
        packages=find_packages(),
        # because of an upload-size limit by PyPI, we're temporarily removing docs from the tarball.
        # Also see MANIFEST.in
        # package_data={'docs': ['*']},
        include_package_data=True,
        classifiers=classifiers,
        python_requires=">=3.7",
        install_requires=install_reqs,
        tests_require=test_reqs,
    )
