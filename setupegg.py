#!/usr/bin/env python
"""
A setup.py script to use setuptools, which gives egg goodness, etc.
"""

from setuptools import setup
with open('setup.py') as s:
    exec(s.read())
