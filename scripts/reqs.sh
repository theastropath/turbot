#!/bin/bash -eu

# Script to determine what packages are installed but not tracked.

TRACKED="$(cat requirements/* | grep -Ev "^-r|^\w*$" | sort -u)"
INSTALLED="$(pip freeze | grep -Ev "turbot|^#|^-e" | sort -u)"
diff -u <(echo "$TRACKED") <(echo "$INSTALLED")
