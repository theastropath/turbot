<img align="right" src="https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png" />

# Turbot

[![build][build-badge]][build]
[![pypi][pypi-badge]][pypi]
[![python][python-badge]][python]
[![codecov][codecov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for everything _Animal Crossing: New Horizons_.

![screenshot](https://user-images.githubusercontent.com/1903876/80841531-787c2f00-8bb4-11ea-8975-cc619b978635.png)

## 🤖 Running the bot

First install `turbot` using [`pip`](https://pip.pypa.io/en/stable/):

```shell
pip install turbot
```

Then you must configure two things:

1. Your Discord bot token.
2. The list of channels you want `turbot` to monitor.

To provide your Discord bot token either set an environment variable named
`TURBOT_TOKEN` to the token or paste it into a file named `token.txt`.

For the list of channels you can provide channel names on the command line using
any number of `--channel "name"` options. Alternatively you can create a file
named `channels.txt` where each line of the file is a channel name. You can
also specify them via the environment variable `TURBOT_CHANNELS` as a semicolon
delimited string, for example: `export TURBOT_CHANNELS="some;channels;here"`.

More usage help can be found by running `turbot --help`.

## 📱 Using the bot

Once you've connected the bot to your server, you can interact with it over
Discord via the following commands in any of the authorized channels.

- `!about`: Get information about Turbot
- `!help`: Provides detailed help about all of the following commands

### 🤔 User Preferences

These commands allow users to set their preferences. These preferences are used
to make other commands more relevant, for example by converting times to the
user's preferred timezone.

- `!info`: Get a user's information
- `!pref`: Set a user preference; use command to get a list of available options

### 💸 Turnips

These commands help users buy low and sell high in the stalk market.

- `!best`: Look for the current best sell or buy
- `!buy`: Save a buy price
- `!clear`: Clear your price data
- `!graph`: Graph price data
- `!history`: Get price history
- `!lastweek`: Get graph for last week's price data
- `!oops`: Undo the last price data
- `!predict`: Predict your price data for the rest of the week
- `!reset`: Reset all users' data
- `!sell`: Save a sell price

### 🐟 Fish & 🐞 Bugs

Provides users with information on where and when to catch critters.

- `!bugs`: Get information on bugs
- `!fish`: Get information on fish
- `!new`: Get information on newly available fish and bugs

### 🦴 Fossils, 🖼️ Art, & 🎶 Songs

When a community of users tracks collectables and trades them between each
other, everyone finishes collecting everything in the game so much more quickly
than they would on their own.

These commands can also help users tell fake art from real art.

- `!art`: Get information on an art piece
- `!collect`: Make something as collected
- `!collected`: Show the things you've collected so far
- `!count`: Count the number of collected things you have
- `!needed`: Find out what collectables are needed by you and others
- `!search`: Search for someone who needs a fossil or art
- `!uncollect`: Remove something from your collection
- `!uncollected`: Get a list of things that you haven't collected yet

### ⚛️ Heroku Support

Turbot supports deployment to Heroku out of the box.
See our documentation on [Heroku Support](HEROKU.md) for details.

---

[MIT][mit] © [TheAstropath][theastropath], [lexicalunit][lexicalunit] et [al][contributors]

[black-badge]:      https://img.shields.io/badge/code%20style-black-000000.svg
[black]:            https://github.com/psf/black
[build-badge]:      https://github.com/theastropath/turbot/workflows/build/badge.svg
[build]:            https://github.com/theastropath/turbot/actions
[codecov-badge]:    https://codecov.io/gh/theastropath/turbot/branch/master/graph/badge.svg
[codecov]:          https://codecov.io/gh/theastropath/turbot
[contributors]:     https://github.com/theastropath/turbot/graphs/contributors
[lexicalunit]:      http://github.com/lexicalunit
[mit-badge]:        https://img.shields.io/badge/License-MIT-yellow.svg
[mit]:              https://opensource.org/licenses/MIT
[python-badge]:     https://img.shields.io/badge/python-3.7+-blue.svg
[python]:           https://www.python.org/
[theastropath]:     https://github.com/theastropath
[pypi]:             https://pypi.org/project/turbot/
[pypi-badge]:       https://img.shields.io/pypi/v/turbot
