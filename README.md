<img align="right" src="turbot.png" />

# Turbot

[![build][build-badge]][build]
[![codecov][codecov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for everything _Animal Crossing: New Horizons_.

![screenshot](https://user-images.githubusercontent.com/1903876/80298832-e784fe00-8744-11ea-8c0f-dbbf81bb5fb7.png)

## 🤖 Running the bot

1. Go to the root directory of this repository.
2. Create a file named `config/token.txt` and paste your Discord bot token into it. Alternatively you can set the token via the environment variable `TURBOT_TOKEN`.
3. Create a file named `config/channels.txt` and paste a list of channels you would like Turbot to run in. Put each channel name on a new line. Alternatively you can provide channel names on the command line using any number of `--channel "name"` options.
4. Ensure that you have the proper Python 3 dependencies installed: `pip install -r requirements.txt`.
5. Install the application: `python setup.py install`.
6. Run the application: `turbot`.
7. For more information on supported features run: `turbot --help`.

## 📱 Using the bot

Once you've connected the bot to your server, you can interact with it over Discord via the following commands in any of the authorized channels.

- `!help` - Provides detailed help about all of the following commands.

### 🤔 User Preferences

These commands allow users to set their preferences. These preferences are used to make other commands more relevant, for example by converting times to the user's preferred timezone.

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

### 🦴 Fossils

When a community of users tracks fossils and trades needed fossils between each other, everyone finishes collecting all the available fossils in the game so much more quickly than they would on their own.

- `!allfossils`
- `!collect`
- `!collectedfossils`
- `!fossilcount`
- `!fossilsearch`
- `!listfossils`
- `!neededfossils`
- `!uncollect`

### 🐟 Fish and Bugs

Provides users with information on where and when to catch critters.

- `!bugs`
- `!fish`
- `!new`

### 🖼️ Art

Helps users tell fake art from real art and tracks your collection.

- `!art`

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
[theastropath]:     https://github.com/theastropath
