<img align="right" src="turbot.png" />

# Turbot

[![build][build-badge]][build]
[![codecov][codecov-badge]][codecov]
[![black][black-badge]][black]
[![mit][mit-badge]][mit]

A Discord bot for tracking _Animal Crossing: New Horizons_ turnip prices and fossil collections.

![screenshot](https://user-images.githubusercontent.com/1903876/80298832-e784fe00-8744-11ea-8c0f-dbbf81bb5fb7.png)

## Usage

1. Go to the root directory of this repository.
2. Create a file named `config/token.txt` and paste your Discord bot token into it. Alternatively you can set the token via the environment variable `TURBOT_TOKEN`.
3. Create a file named `config/channels.txt` and paste a list of channels you would like Turbot to run in. Put each channel name on a new line.
4. Ensure that you have the proper Python 3 dependencies installed: `pip install -r requirements.txt`.
5. Install the application: `python setup.py install`.
6. Run the application: `turbot`.

[black-badge]:      https://img.shields.io/badge/code%20style-black-000000.svg
[black]:            https://github.com/psf/black
[build-badge]:      https://github.com/theastropath/turbot/workflows/build/badge.svg
[build]:            https://github.com/theastropath/turbot/actions
[codecov-badge]:    https://codecov.io/gh/theastropath/turbot/branch/master/graph/badge.svg
[codecov]:          https://codecov.io/gh/theastropath/turbot
[mit-badge]:        https://img.shields.io/badge/License-MIT-yellow.svg
[mit]:              https://opensource.org/licenses/MIT
