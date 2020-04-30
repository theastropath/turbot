<img align="right" src="https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png" />

# Turbot

[![build][build-badge]][build]
[![python][python-badge]][python]
[![codecov][codecov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for everything _Animal Crossing: New Horizons_.

![screenshot](https://user-images.githubusercontent.com/1903876/80298832-e784fe00-8744-11ea-8c0f-dbbf81bb5fb7.png)

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

- `!help` - Provides detailed help about all of the following commands.

### 🤔 User Preferences

These commands allow users to set their preferences. These preferences are used
to make other commands more relevant, for example by converting times to the
user's preferred timezone.

- `!hemisphere`
- `!timezone`

### 💸 Turnips

These commands help users buy low and sell high in the stalk market.

- `!bestsell`
- `!buy`
- `!clear`
- `!graph`
- `!history`
- `!lastweek`
- `!oops`
- `!predict`
- `!reset`
- `!sell`
- `!turnippattern`

### 🐟 Fish and Bugs

Provides users with information on where and when to catch critters.

- `!bugs`
- `!fish`
- `!new`

### 🦴 Fossils & 🖼️ Art

When a community of users tracks collectables and trades them between each
other, everyone finishes collecting everything in the game s much more quickly
than they would on their own.

These commands can also help users tell fake art from real art.

- `!allart`
- `!allfossils`
- `!art`
- `!collect`
- `!collected`
- `!count`
- `!neededfossils`
- `!search`
- `!uncollect`
- `!uncollected`

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
