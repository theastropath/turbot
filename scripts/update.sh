#!/bin/bash -eu

# Script to update all dependencies.

pip list --outdated --format=freeze |
    grep -v '^\-e' |
    cut -d = -f 1  |
    xargs -n1 pip install -U --ignore-installed
