import argparse

from pathlib import Path

import yaml

ERR_MESSAGE = "File {} contains theano-pymc version {} while requirements.txt has version {}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    with open("requirements.txt") as fd:
        for line in fd:
            if line.startswith("theano-pymc"):
                _, theano_pin = line.strip().split("==")
                break
        else:
            raise RuntimeError("requirements.txt does not contain theano-pymc")

    for path in args.paths:
        with open(str(path)) as fd:
            deps = yaml.safe_load(fd)["dependencies"]
        for dep in deps:
            if dep.startswith("theano-pymc"):
                _, version = dep.split("=")
                assert version == theano_pin, ERR_MESSAGE.format(str(path), version, theano_pin)
