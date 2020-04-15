# Turbot

A Discord bot for tracking _Animal Crossing: New Horizons_ turnip prices and fossil collections.

## Usage

1. Go to the directory where you have `turbot.py` located:
2. Create a file named `token.txt` and paste your Discord bot token into it.
3. Create a file named `channels.txt` and paste a list of channels you would like Turbot to run in. Put each channel name on a new line.
4. Ensure that you have the proper Python 3 dependencies installed: `pip install -r requirements.txt`.
5. Install the application: `python setup.py install`.
6. Run the application: `turbot`.

## Development

This bot is developed for Python 3.7+ and it is built on top of the [`discord.py`](https://github.com/Rapptz/discord.py) library.

To install development dependencies (which also includes the production dependencies) use: `pip install -r requirements/dev.txt`. This will allow you to run [PyTest](https://docs.pytest.org/en/latest/) tests locally.

### Running tests

Run the tests with [tox](https://tox.readthedocs.io/en/latest/):

```shell
$ tox
```

### Installing the application

Install in development mode using [setuptools](https://setuptools.readthedocs.io/en/latest/):

```shell
$ python setup.py develop
```

### Running the application

Make sure that you have created the required "token.txt" and "channels.txt" as described above in the usage section. Setuptools will have installed an entry point for you, so run it:

```bash
$ turbot
```

### Migration

The older file format for turbot data was individual files per user stored in the `prices` and `fossils` directories. If you have existing data in this format and wish to automatically migrate it to the newest format, use the provided migration script:

```bash
$ migrate
```

The script can be run multiple times without corrupting or duplicating data. Any existing new format data will be merged with the old format data and saved.

### Formatting and linting

Codebase consistency is maintained by the industry standard too [black](https://black.readthedocs.io/en/stable/). For linting we use [flake8](https://flake8.pycqa.org/en/latest/) with the [flake8-black](https://pypi.org/project/flake8-black/) support plugin. Imports are kept in order by [isort](https://timothycrosley.github.io/isort/). The included test suite can run these tools against the codebase and report on any errors:

```bash
$ tox -- -k codebase
```
