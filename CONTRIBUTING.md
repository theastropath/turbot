# Contributing

This bot is developed for Python 3.7 and it is built on top of the [`discord.py`](https://github.com/Rapptz/discord.py) library.

It uses [`poetry`](usage) to manage dependencies. To install development dependencies use: `poetry install`. This will allow you to run [PyTest](https://docs.pytest.org/en/latest/) and the included scripts.

You can install `poetry` with `pip`:

```shell
pip install poetry
```

## Running tests

We use [tox](https://tox.readthedocs.io/en/latest/) to manage test execution. It can be installed with `pip`:

```shell
pip install tox
```

Then run all tests with:

```shell
tox
```

or a specific set of tests, like all tests pertaining to fossils for example:

```shell
tox -- -k fossil
```

After running the _full test suite_ you can view code coverage reports in the newly created `htmlcov` directory.

```shell
open htmlcov/index.htmlcov
```

## Installing the application

Install in development mode via `poetry`:

```shell
poetry install
```

## Running the application

Make sure that you have created the required "token.txt" and "channels.txt" files or else provide them via the command line.

```shell
poetry run turbot --help
```

## Formatting and linting

Codebase consistency is maintained by the industry standard [black][black]. For linting we use [flake8](https://flake8.pycqa.org/en/latest/) with configuration to work alongside the formatter. Imports are kept in order by [isort](https://timothycrosley.github.io/isort/). The included test suite can run these tools against the codebase and report on any errors:

```shell
tox -- -k codebase
```

## Updating application data

Data on art, bugs, and fish comes directly from [the Animal Crossing fandom page][wiki] and then compiled into a csv file in the package's `data` directory. The scripts to do this are included in the `scripts` directory. Run them to fetch the latest data:

```shell
poetry run scripts/update_art_data.py
poetry run scripts/update_bugs_data.py
poetry run scripts/update_fish_data.py
```

## Updating baseline figures

We use [pytest-mpl](https://github.com/matplotlib/pytest-mpl) to verify generated graphs in our test suite. Generating the baseline images is a bit of a process. From the root of this repository do:

```shell
poetry run pytest -k figures --mpl-generate-path=tests/baseline
```

## Updating test snapshots

We also use [pytest-snapshot](https://github.com/joseph-roitman/pytest-snapshot) to generate and test against snapshots. To generate new snapshots pass `--snapshot-update` to your `pytest` command. For example, from the root of this repository:

```shell
poetry run pytest -k your_test_function --snapshot-update
```

## Release process

To release a new version of `turbot`, use `poetry`:

```shell
poetry version [major|minor|patch]
git commit -am "Release vM.M.P"
git push
poetry publish
```

You can get the `M.M.P` version numbers from `pyproject.toml` after you've run the `poetry version` command. On a *NIX shell you could also get automatically it like so:

```shell
cat pyproject.toml | grep "^version" | cut -d= -f2 | sed 's/"//g;s/ //g;s/^/v/;'
```

When you use the `poetry publish` command you will be prompted for your [PyPI](https://pypi.org/) credentials.

After publishing you can view the package at https://pypi.org/project/turbot/ to see that everything looks good.

[black]:            https://github.com/psf/black
[wiki]:             https://animalcrossing.fandom.com/
