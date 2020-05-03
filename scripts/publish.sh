#!/bin/bash

set -eu

usage() {
    echo "usage: ${0##*/} [minor|major|patch] [-h|--help]" 1>&2
    echo "Bumps the minor, major, or patch version and releases the package." 1>&2
    exit 1
}

if echo "$*" | grep -Eq -- '--help\b|-h\b' || [[ -z $1 ]]; then
    usage
fi

KIND="$1"

if [[ "$KIND" != "major" && "$KIND" != "minor" && "$KIND" != "patch" ]]; then
    usage
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ $BRANCH != "master" ]]; then
    echo "error: you must be on the master branch to publish" 1>&2
    exit 1
fi

CHANGES="$(git status -su)"
if [[ -n "$CHANGES" ]]; then
    echo "error: can not publish when there are uncomitted changes" 1>&2
    exit 1
fi

# bump the version in pyproject.toml
poetry version "$KIND"

# install the new version
poetry install

# run once to rebuild the snapshot
poetry run pytest -k test_on_message_about --snapshot-update 2>/dev/null 1>&2 || true

# run again to ensure the build is good
poetry run pytest

# fetch the version from pyproject.toml
VERSION="$(grep "^version" < pyproject.toml | cut -d= -f2 | sed 's/"//g;s/ //g;s/^/v/;')"

# build the release
poetry build

# commit changes
git commit -am "Release $VERSION"

# push changes to origin/master
git push origin master

# publish the release; assumes you've set up non-interactive publishing by
# previously having run: `poetry config pypi-token.pypi "YOUR-PYPI-TOKEN-GOES-HERE"`
if ! poetry publish -n; then
    echo "error: publish command failed, see log for details" 1>&2
    git reset --hard HEAD~1
fi
