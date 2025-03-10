"""Parse the GitHub action log for test times.

Taken from https://github.com/pymc-labs/pymc-marketing/tree/main/scripts/slowest_tests/extract-slow-tests.py

"""

import re
import sys

from pathlib import Path

start_pattern = re.compile(r"==== slow")
separator_pattern = re.compile(r"====")
time_pattern = re.compile(r"(\d+\.\d+)s ")


def extract_lines(lines: list[str]) -> list[str]:
    times = []

    in_section = False
    for line in lines:
        detect_start = start_pattern.search(line)
        detect_end = separator_pattern.search(line)

        if detect_start:
            in_section = True

        if in_section:
            times.append(line)

        if not detect_start and in_section and detect_end:
            break

    return times


def trim_up_to_match(pattern, string: str) -> str:
    match = pattern.search(string)
    if not match:
        return ""

    return string[match.start() :]


def trim(pattern, lines: list[str]) -> list[str]:
    return [trim_up_to_match(pattern, line) for line in lines]


def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def format_times(times: list[str]) -> list[str]:
    return (
        trim(separator_pattern, times[:1])
        + trim(time_pattern, times[1:-1])
        + [strip_ansi(line) for line in trim(separator_pattern, times[-1:])]
    )


def read_lines_from_stdin():
    return sys.stdin.read().splitlines()


def read_from_file(file: Path):
    """For testing purposes."""
    return file.read_text().splitlines()


def main(read_lines):
    lines = read_lines()
    times = extract_lines(lines)
    parsed_times = format_times(times)
    print("\n".join(parsed_times))  # noqa: T201


if __name__ == "__main__":
    read_lines = read_lines_from_stdin
    main(read_lines)
