<img align="right" src="https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png" />

# Turbot

[![build][build-badge]][build]
[![pypi][pypi-badge]][pypi]
[![python][python-badge]][python]
[![codecov][codecov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for everything _Animal Crossing: New Horizons_.

[![add-bot][add-img]][add-bot]

## 📱 Using Turbot

Once you've connected the bot to your server, you can interact with it over
Discord via the following commands in any of the authorized channels.

- `!about`: Get information about Turbot
- `!help`: Provides detailed help about all of the following commands

### 💸 Turnips

![predictions](https://user-images.githubusercontent.com/1903876/82263275-63730000-9917-11ea-94d1-38661784097c.png)

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

### ℹ️ User Preferences

![user-info](https://user-images.githubusercontent.com/1903876/82263272-61a93c80-9917-11ea-9e8c-ded5eb1f652e.png)

These commands allow users to set their preferences. These preferences are used
to make other commands more relevant, for example by converting times to the
user's preferred timezone.

- `!info`: Get a user's information
- `!pref`: Set a user preference; use command to get a list of available options

### 📮 Collectables

![collecting](https://user-images.githubusercontent.com/1903876/82263264-5f46e280-9917-11ea-9c1e-90d4077013ca.png)

When a community of users tracks collectables and trades them between each
other, everyone finishes collecting everything in the game so much more quickly
than they would on their own. Turbot supports collecting:

- 🦴 Fossils
- 🐞 Bugs
- 🐟 Fish
- 🖼️ Art
- 🎶 Songs

#### 📈 Managing your Collection

- `!collect`: Mark something as collected
- `!collected`: Show the things you've collected so far
- `!count`: Count the number of collected things you have
- `!needed`: Find out what collectables are needed by you and others
- `!search`: Search for someone who needs something you're looking to give away
- `!uncollect`: Remove something from your collection
- `!uncollected`: Get a list of things that you haven't collected yet

#### 🤔 Helper Utilities

Some collections require additional support such as the `!art` command that
helps users tell fake art from real art. The `!bugs` and `!fish` commands
tell users when and where to catch those critters. These commands also know what
you've already collected and will tailor their responses to the user.

- `!art`: Get information on an art piece
- `!bugs`: Get information on bugs
- `!fish`: Get information on fish
- `!new`: Get information on newly available fish and bugs

### 👑 Administration

- `!authorize`: Configure which channels Turbot can operate in

## 🤖 Running Turbot

First install `turbot` using [`pip`](https://pip.pypa.io/en/stable/):

```shell
pip install turbot
```

Then you must configure two things:

1. Your Discord bot token.
2. The list of channels you want `turbot` to monitor. (Default: All channels)

To provide your Discord bot token either set an environment variable named
`TURBOT_TOKEN` to the token or paste it into a file named `token.txt`.

For the list of channels you can provide channel names on the command line using
any number of `--channel "name"` options. Alternatively you can create a file
named `channels.txt` where each line of the file is a channel name. You can
also specify them via the environment variable `TURBOT_CHANNELS` as a semicolon
delimited string, for example: `export TURBOT_CHANNELS="some;channels;here"`.
You can also leave this unspecified and Turbot will operate within all channels,
then you can specify a smaller set of channels using the `!authorize` command.

By default Turbot will use sqlite3 as its database. You can however choose to
use another database by providing a [SQLAlchemy Connection URL][db-url]. This
can be done via the `--database-url` command line option or the environment
variable `TURBOT_DB_URL`. Note that, at the time of this writing, Turbot is only
tested against sqlite3 and PostgreSQL.

More usage help can be found by running `turbot --help`.

## ⚛️ Heroku Support

Turbot supports deployment to Heroku out of the box. All you need is your
Discord bot token and you're ready to go! Just click the Deploy to Heroku
button, below.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)][deploy]

For more details see [our documentation on Heroku support](HEROKU.md).

## 🐳 Docker Support

You can also run Turbot via docker. See
[our documentation on Docker Support](DOCKER.md) for help.

---

[MIT][mit] © [TheAstropath][theastropath], [lexicalunit][lexicalunit] et [al][contributors]

[add-bot]:          https://discordapp.com/api/oauth2/authorize?client_id=699774176155926599&permissions=247872&scope=bot
[add-img]:          https://user-images.githubusercontent.com/1903876/82262797-71745100-9916-11ea-8b65-b3f656115e4f.png
[black-badge]:      https://img.shields.io/badge/code%20style-black-000000.svg
[black]:            https://github.com/psf/black
[build-badge]:      https://github.com/theastropath/turbot/workflows/build/badge.svg
[build]:            https://github.com/theastropath/turbot/actions
[codecov-badge]:    https://codecov.io/gh/theastropath/turbot/branch/master/graph/badge.svg
[codecov]:          https://codecov.io/gh/theastropath/turbot
[contributors]:     https://github.com/theastropath/turbot/graphs/contributors
[db-url]:           https://docs.sqlalchemy.org/en/latest/core/engines.html
[deploy]:           https://heroku.com/deploy
[lexicalunit]:      http://github.com/lexicalunit
[mit-badge]:        https://img.shields.io/badge/License-MIT-yellow.svg
[mit]:              https://opensource.org/licenses/MIT
[pypi-badge]:       https://img.shields.io/pypi/v/turbot
[pypi]:             https://pypi.org/project/turbot/
[python-badge]:     https://img.shields.io/badge/python-3.7+-blue.svg
[python]:           https://www.python.org/
[theastropath]:     https://github.com/theastropath
