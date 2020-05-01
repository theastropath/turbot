<img align="right" src="https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png" />

# Turbot

[![build][build-badge]][build]
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
named `channels.txt` where each line of the file is a channel name.

More usage help can be found by running `turbot --help`.

## 📱 Using the bot

Once you've connected the bot to your server, you can interact with it over
Discord via the following commands in any of the authorized channels.

- `!help` - Provides detailed help about all of the following commands
- `!about` - Get information about Turbot

### 🤔 User Preferences

These commands allow users to set their preferences. These preferences are used
to make other commands more relevant, for example by converting times to the
user's preferred timezone.

- `!friend`: Set your friend code
- `!fruit`: Set your native fruit
- `!hemisphere`: Set your island hemisphere
- `!info`: Get a user's information
- `!island`: Set your island name
- `!nickname`: Set your nickname
- `!timezone`: Set your island timezone

### 💸 Turnips

These commands help users buy low and sell high in the stalk market.

- `!bestsell`: Look for the best buy
- `!buy`: Save a buy price
- `!clear`: Clear your price data
- `!graph`: Graph price data
- `!history`: Get price history
- `!lastweek`: Get graph for last week's price data
- `!oops`: Undo the last price data
- `!predict`: Predict your price data for the rest of the week
- `!reset`: Reset all users' data
- `!sell`: Save a sell price
- `!turnippattern`: Determine your turnip price pattern for the week

### 🐟 Fish and Bugs

Provides users with information on where and when to catch critters.

- `!bugs`: Get information on bugs
- `!fish`: Get information on fish
- `!new`: Get information on newly available fish and bugs

### 🦴 Fossils & 🖼️ Art

When a community of users tracks collectables and trades them between each
other, everyone finishes collecting everything in the game s much more quickly
than they would on their own.

These commands can also help users tell fake art from real art.

- `!allfossils`: Get a list of all possible fossils
- `!art`: Get information on an art piece
- `!collect`: Collect fossils or art
- `!collected`: Show collected fossils and art
- `!count`: Count the number of collected fossils and art
- `!neededfossils`: Get what fossils are needed by users
- `!search`: Search for someone who needs a fossil or art
- `!uncollect`: Remove a fossil or art from your collection
- `!uncollected`: Get fossils and art not yet collected

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
