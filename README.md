# Turbot

[![build][build-badge]][build]
[![codecov][codcov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for tracking _Animal Crossing: New Horizons_ turnip prices and fossil collections.

## Usage

1. Go to the root directory of this repository.
2. Create a file named `config/token.txt` and paste your Discord bot token into it.
3. Create a file named `config/channels.txt` and paste a list of channels you would like Turbot to run in. Put each channel name on a new line.
4. Ensure that you have the proper Python 3 dependencies installed: `pip install -r requirements.txt`.
5. Install the application: `python setup.py install`.
6. Run the application: `turbot`.

## Development

This bot is developed for Python 3.7+ and it is built on top of the [`discord.py`](https://github.com/Rapptz/discord.py) library.

To install development dependencies (which also includes the production dependencies) use: `pip install -r requirements/dev.txt`. This will allow you to run [PyTest](https://docs.pytest.org/en/latest/) tests locally.

### Running tests

Run the tests with [tox](https://tox.readthedocs.io/en/latest/):

```shell
tox
```

### Installing the application

Install in development mode using [setuptools](https://setuptools.readthedocs.io/en/latest/):

```shell
python setup.py develop
```

### Running the application

Make sure that you have created the required "token.txt" and "channels.txt" files as described above in the usage section. Setuptools will have installed an entry point for you, so run it:

```shell
turbot
```

### Migration

The older file format for turbot data was individual files per user stored in the `prices` and `fossils` directories. If you have existing data in this format and wish to automatically migrate it to the newest format, use the provided migration script:

```shell
migrate
```

The script can be run multiple times without corrupting or duplicating data. Any existing new format data will be merged with the old format data and saved.

### Formatting and linting

Codebase consistency is maintained by the industry standard [black][black]. For linting we use [flake8](https://flake8.pycqa.org/en/latest/) with configuration to work alongside the formatter. Imports are kept in order by [isort](https://timothycrosley.github.io/isort/). The included test suite can run these tools against the codebase and report on any errors:

```shell
tox -- -k codebase
```

### Updating fish data

Data on fishes is captured from [the Animal Crossing fandom page](https://animalcrossing.fandom.com/wiki/Fish_(New_Horizons)) and then compiled into a csv file in the package's `data` directory. The script to do this is included in the `scripts` directory. Run it to fetch the latest data:

```shell
./scripts/update_fish_data.py
```

Note that you must have development requirements installed to run this script.

[black-badge]:      https://img.shields.io/badge/code%20style-black-000000.svg
[black]:            https://github.com/psf/black
[build-badge]:      https://github.com/theastropath/turbot/workflows/build/badge.svg
[build]:            https://github.com/theastropath/turbot/actions
[codecov-badge]:    https://codecov.io/gh/theastropath/turbot/branch/master/graph/badge.svg
[codecov]:          https://codecov.io/gh/theastropath/turbot
[mit-badge]:        https://img.shields.io/badge/License-MIT-yellow.svg
[mit]:              https://opensource.org/licenses/MIT
