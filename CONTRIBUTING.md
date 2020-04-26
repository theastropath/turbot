# Contributing

This bot is developed for Python 3.7+ and it is built on top of the [`discord.py`](https://github.com/Rapptz/discord.py) library.

To install development dependencies (which also includes the production dependencies) use: `pip install -r requirements/dev.txt`. This will allow you to run [PyTest](https://docs.pytest.org/en/latest/) tests locally.

## Running tests

Run all these tests with [tox](https://tox.readthedocs.io/en/latest/):

```shell
tox
```

or a specific set of tests, like all tests pertaining to fossils for example:

```shell
tox -- -k fossil
```

After running the _full test suite_ you can view code coverage reports in the newly created `htmlcov` directory.

## Installing the application

Install in development mode using [setuptools](https://setuptools.readthedocs.io/en/latest/):

```shell
python setup.py develop
```

## Running the application

Make sure that you have created the required "token.txt" and "channels.txt" files as described above in the usage section. Setuptools will have installed an entry point for you, so run it to get usage help:

```shell
turbot --help
```

## Formatting and linting

Codebase consistency is maintained by the industry standard [black][black]. For linting we use [flake8](https://flake8.pycqa.org/en/latest/) with configuration to work alongside the formatter. Imports are kept in order by [isort](https://timothycrosley.github.io/isort/). The included test suite can run these tools against the codebase and report on any errors:

```shell
tox -- -k codebase
```

## Updating fish and bugs data

Data on fish and bugs comes directly from [the Animal Crossing fandom page][wiki] and then compiled into a csv file in the package's `data` directory. The scripts to do this are included in the `scripts` directory. Run them to fetch the latest data:

```shell
./scripts/update_bugs_data.py
./scripts/update_fish_data.py
```

Note that you must have development requirements installed to run these scripts.

## Updating baseline figures

We use [pytest-mpl](https://github.com/matplotlib/pytest-mpl) to verify generated graphs in our test suite. Generating the baseline images is a bit of a process. From the root of this repository do:

```shell
python setup.py develop # or `python setup.py install --force` if you have it installed locally
pytest -k figures --mpl-generate-path=tests/baseline
```

## Updating test snapshots

We also use [pytest-snapshot](https://github.com/joseph-roitman/pytest-snapshot) to generate and test against snapshots. To generate new snapshots pass `--snapshot-update` to your `pytest` command. For example, from the root of this repository:

```shell
python setup.py develop # or `python setup.py install --force` if you have it installed locally
pytest -k your_test_function --snapshot-update
```

[black]:            https://github.com/psf/black
[wiki]:             https://animalcrossing.fandom.com/
