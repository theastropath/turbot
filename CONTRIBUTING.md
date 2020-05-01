# Contributing

This bot is developed for Python 3.7+ and it is built on top of the
[`discord.py`](https://github.com/Rapptz/discord.py) library.

It uses [`poetry`](usage) to manage dependencies. To install development
dependencies use: `poetry install`. This will allow you to run
[PyTest](https://docs.pytest.org/en/latest/) and the included scripts.

You can install `poetry` with `pip`:

```shell
pip install poetry
```

## Installing the application

Install in development mode via `poetry`:

```shell
poetry install
```

> **Note:** On some systems such as most Linux ones you may also need to install
> `python3-venv` using your system's package manager as `poetry` depends on it.

## Running the application

Make sure that you have created the required "token.txt" and "channels.txt"
files or else provide them via the command line.

```shell
poetry run turbot --help
```

## Running tests

We use [tox](https://tox.readthedocs.io/en/latest/) to manage test execution.
It can be installed with `pip`:

```shell
pip install tox
```

Whenever you ran the `poetry install` command for the first time it created a
virtual environment for you based on which python environment you installed
`poetry` into.

Our `tox` configuration has tests set to run against multiple python
environments. So you will need to manage multiple installs of python to be able
to support testing in multiple environments. Thankfully
[`pyenv`](https://github.com/pyenv/pyenv) and
[`tox-pyenv`](https://pypi.org/project/tox-pyenv/) make this straightforward and
seamless.

First [install `pyenv`](https://github.com/pyenv/pyenv#installation) on your
machine. There are various methods to do this, some easier than others. On macOS
I'd use [Homebrew](https://brew.sh/) and for other systems I, and the author of
`pyenv`, recommend
[`pyenv-installer`](https://github.com/pyenv/pyenv-installer).

Now install [`tox-pyenv`](https://pypi.org/project/tox-pyenv/) so that `tox` can
automatically find your environments:

```shell
pip install tox-pyenv
```

Now let's create the python environments you'll need. The following commands
will do this on any *NIX system. For other systems hopefully what I'm doing here
is instructive. Details on each command are given with inline comments.

```shell
# install a plugin that allows pyenv to know how to fetch the latest versions
git clone https://github.com/momo-lab/xxenv-latest.git "$(pyenv root)"/plugins/xxenv-latest

# for each python environment in the tox configuration, create it using pyenv
# this step takes a while, but you will only need to do this setup once
tox -l | while read -r py; do
    # translate a tox env name like "py37" into a number like "37"
    number="$(echo "$py" | sed "s/^py//")"

    # translate a number like "37" into a python version like "3.7"
    version="${number:0:1}.${number:1:1}"

    # install the latest python interpreter for that version
    pyenv latest install "$version"
done

# configure turbot to use the interpreters we just installed
pyenv local $(pyenv versions --bare | tr '\n' ' ')
```

The above script, **at the time of this writing**, amounts to running the
following three commands:

```shell
pyenv install 3.7.6
pyenv install 3.8.1
pyenv local 3.7.6 3.8.1
```

But over time newer versions of python will become available and the above
script should automatically handle that for you.

> **Note:** If your `pyenv install` command fails, please read
> [this](https://github.com/pyenv/pyenv/wiki/common-build-problems) help
> documentation provided by `pyenv`. You probably need to install some
> prerequisites on your system using your system's package manager.

Now you should be able to run the entire test suite against all python
environments with:

```shell
tox
```

Or a specific set of tests and environment. For example all tests pertaining to
fossils using the `py37` environment:

```shell
tox -e py37 -- -k fossil
```

After running the _full test suite_ you can view code coverage reports in the
`coverage` directory broken out by python environment.

```shell
open coverage/py37/index.html
```

## Formatting and linting

Codebase consistency is maintained by the industry standard [black][black]. For
linting we use [flake8](https://flake8.pycqa.org/en/latest/) with configuration
to work alongside the formatter. Imports are kept in order by
[isort](https://timothycrosley.github.io/isort/). The included test suite can
run these tools against the codebase and report on any errors:

```shell
tox -- -k codebase
```

## Updating application data

Data on art, bugs, and fish comes directly from
[the Animal Crossing fandom page][wiki] and then compiled into a csv file in the
package's `data` directory. The scripts to do this are included in the `scripts`
directory. Run them to fetch the latest data:

```shell
poetry run scripts/update_art_data.py
poetry run scripts/update_bugs_data.py
poetry run scripts/update_fish_data.py
```

## Updating baseline figures

We use [pytest-mpl](https://github.com/matplotlib/pytest-mpl) to verify
generated graphs in our test suite. Generating the baseline images is a bit of a
process. From the root of this repository do:

```shell
poetry run pytest -k figures --mpl-generate-path=tests/baseline
```

## Updating test snapshots

We also use [pytest-snapshot](https://github.com/joseph-roitman/pytest-snapshot)
to generate and test against snapshots. To generate new snapshots pass
`--snapshot-update` to your `pytest` command. For example, from the root of this
repository:

```shell
poetry run pytest -k your_test_function --snapshot-update
```

## Release process

To release a new version of `turbot`, use `poetry`:

```shell
poetry version [major|minor|patch]
poetry run pytest -k test_on_message_about --snapshot-update
git commit -am "Release vM.M.P"
git push
poetry build
poetry publish
```

> **Note:** The reason you have to run `pytest` after running the
> `poetry version` command is because we must update some test snapshots as the
> version number can appear in them. Failure to do this will result in a
> failing build on `master`.

You can get the `M.M.P` version numbers from `pyproject.toml` after you've run
the `poetry version` command. On a *NIX shell you could also get automatically
it like so:

```shell
grep "^version" < pyproject.toml | cut -d= -f2 | sed 's/"//g;s/ //g;s/^/v/;'
```

When you use the `poetry publish` command you will be prompted for your
[PyPI](https://pypi.org/) credentials.

After publishing you can view the package at its
[pypi.org project page](https://pypi.org/project/turbot/) to see that everything
looks good.

> **Note:** If you want to do this release process automatically there is a
> *NIX script available in the `scripts` directory to help. To use it you will
> need to have non-interactive `poetry publish` enabled by running:
> `poetry config pypi-token.pypi "YOUR-PYPI-TOKEN-GOES-HERE"`.

[black]:            https://github.com/psf/black
[wiki]:             https://animalcrossing.fandom.com/
