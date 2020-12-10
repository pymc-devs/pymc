import argparse

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    for file_ in args.files:
        with open(file_) as fd:
            doc = yaml.safe_load(fd)
            doc["dependencies"].sort()
        with open(file_, "w") as fd:
            yaml.dump(doc, fd, sort_keys=False)
