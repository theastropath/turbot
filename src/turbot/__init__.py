import inspect
import json
import logging
import random
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from itertools import product
from os import getenv
from os.path import dirname, realpath
from pathlib import Path
from string import Template

import click
import discord
import dunamai as _dunamai
import hupper
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from humanize import naturaltime
from turnips.archipelago import Archipelago
from turnips.plots import plot_models_range
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader

__version__ = _dunamai.get_version(
    "turbot", third_choice=_dunamai.Version.from_any_vcs
).serialize()

matplotlib.use("Agg")

PACKAGE_ROOT = Path(dirname(realpath(__file__)))
RUNTIME_ROOT = Path(".")

# application configuration files
DEFAULT_CONFIG_TOKEN = RUNTIME_ROOT / "token.txt"
DEFAULT_CONFIG_CHANNELS = RUNTIME_ROOT / "channels.txt"

# static application asset data
DATA_DIR = PACKAGE_ROOT / "data"
STRINGS_DATA_FILE = DATA_DIR / "strings.yaml"
FOSSILS_DATA_FILE = DATA_DIR / "fossils.txt"
FISH_DATA_FILE = DATA_DIR / "fish.csv"
BUGS_DATA_FILE = DATA_DIR / "bugs.csv"
ART_DATA_FILE = DATA_DIR / "art.csv"

# persisted user and application data
DB_DIR = RUNTIME_ROOT / "db"
DEFAULT_DB_FOSSILS = DB_DIR / "fossils.csv"
DEFAULT_DB_PRICES = DB_DIR / "prices.csv"
DEFAULT_DB_ART = DB_DIR / "art.csv"
DEFAULT_DB_USERS = DB_DIR / "users.csv"

# temporary application files
TMP_DIR = RUNTIME_ROOT / "tmp"
GRAPHCMD_FILE = TMP_DIR / "graphcmd.png"
LASTWEEKCMD_FILE = TMP_DIR / "lastweek.png"

with open(STRINGS_DATA_FILE) as f:
    STRINGS = load(f, Loader=Loader)

FISH = pd.read_csv(FISH_DATA_FILE)
BUGS = pd.read_csv(BUGS_DATA_FILE)
ART = pd.read_csv(ART_DATA_FILE)

with open(FOSSILS_DATA_FILE) as f:
    FOSSILS_SET = frozenset([line.strip().lower() for line in f.readlines()])
FISH_SET = frozenset(FISH.drop_duplicates(subset="name").name.tolist())
BUGS_SET = frozenset(BUGS.drop_duplicates(subset="name").name.tolist())
ART_SET = frozenset(ART.drop_duplicates(subset="name").name.tolist())
COLLECTABLE_SET = FOSSILS_SET | FISH_SET | BUGS_SET | ART_SET

EMBED_LIMIT = 5  # more embeds in a row than this causes issues

DAYS = {
    "sunday": 0,
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
}


def s(key, **kwargs):
    """Returns a string from data/strings.yaml with subsitutions."""
    data = STRINGS.get(key, "")
    assert data, f"error: missing strings key: {key}"
    return Template(data).substitute(kwargs)


def h(dt):
    """Convertes a datetime to something readable by a human."""
    if hasattr(dt, "tz_convert"):  # pandas-datetime-like objects
        dt = dt.to_pydatetime()
    naive_dt = dt.replace(tzinfo=None)
    return naturaltime(naive_dt)


def humanize_months(row):
    """Generator that humanizes months from row data where each month is a column."""
    ABBR = {
        0: "Jan",
        1: "Feb",
        2: "Mar",
        3: "Apr",
        4: "May",
        5: "Jun",
        6: "Jul",
        7: "Aug",
        8: "Sep",
        9: "Oct",
        10: "Nov",
        11: "Dec",
    }
    months = [
        row["jan"],
        row["feb"],
        row["mar"],
        row["apr"],
        row["may"],
        row["jun"],
        row["jul"],
        row["aug"],
        row["sep"],
        row["oct"],
        row["nov"],
        row["dec"],
    ]
    start = None
    for m, inc in enumerate(months):
        if inc and start is None:
            start = m  # start of a range
        elif not inc and start is None:
            continue  # range hasn't started yet
        elif inc and start is not None:
            continue  # continuance of a range
        else:
            lhs = ABBR[start]
            rhs = ABBR[m - 1]
            if lhs != rhs:
                yield f"{lhs} - {rhs}"  # previous element ended a range
            else:
                yield f"{lhs}"  # captures a lone element
            start = None
    if start == 0:
        yield "the entire year"  # capture total range
    elif start is not None:
        lhs = ABBR[start]
        rhs = ABBR[11]
        if lhs != rhs:
            yield f"{lhs} - {rhs}"  # capture a trailing range
        else:
            yield f"{lhs}"  # captures a trailing lone element


def discord_user_from_name(channel, name):
    """Returns the discord user from the given channel and name."""
    if name is None:
        return None
    lname = name.lower()
    members = channel.members
    return next(filter(lambda member: lname in str(member).lower(), members), None)


def discord_user_from_id(channel, user_id):
    """Returns the discord user from the given channel and user id."""
    iid = int(user_id)
    members = channel.members
    return next(filter(lambda member: iid == member.id, members), None)


def discord_user_name(channel, name_or_id):
    """Returns the discord user name from the given channel and name or id."""
    if not name_or_id:
        return None
    user = (
        discord_user_from_id(channel, name_or_id)
        if isinstance(name_or_id, int) or name_or_id.isdigit()
        else discord_user_from_name(channel, name_or_id)
    )
    return str(user) if user else None


def discord_user_id(channel, name):
    """Returns the discord user id name from the given channel and name."""
    if not name:
        return None
    return getattr(discord_user_from_name(channel, name), "id", None)


def is_turbot_admin(channel, user_or_member):
    """Checks to see if given user or member has the Turbot Admin role on this server."""
    member = (
        user_or_member
        if hasattr(user_or_member, "roles")  # members have a roles property
        else channel.guild.get_member(user_or_member.id)  # but users don't
    )
    return any(role.name == "Turbot Admin" for role in member.roles) if member else False


class Turbot(discord.Client):
    """Discord turnip bot"""

    def __init__(
        self,
        token="",
        channels=[],
        prices_file=DEFAULT_DB_PRICES,
        art_file=DEFAULT_DB_ART,
        fossils_file=DEFAULT_DB_FOSSILS,
        users_file=DEFAULT_DB_USERS,
        log_level=None,
    ):
        if log_level:  # pragma: no cover
            logging.basicConfig(level=log_level)
        super().__init__()
        self.token = token
        self.channels = channels
        self.prices_file = prices_file
        self.art_file = art_file
        self.fossils_file = fossils_file
        self.users_file = users_file
        self.base_prophet_url = "https://turnipprophet.io/?prices="  # TODO: configurable?
        self._prices_data = None  # do not use directly, load it from load_prices()
        self._art_data = None  # do not use directly, load it from load_art()
        self._fossils_data = None  # do not use directly, load it from load_fossils()
        self._users_data = None  # do not use directly, load it from load_users()
        self._last_backup_filename = None

    def run(self):  # pragma: no cover
        super().run(self.token)

    def save_prices(self, data):
        """Saves the given prices data to csv file."""
        data.to_csv(self.prices_file, index=False)  # persist to disk
        self._prices_data = data  # in-memory optimization

    def last_backup_filename(self):
        """Return the name of the last known backup file for prices or None if unknown."""
        return self._last_backup_filename

    def backup_prices(self, data):
        """Backs up the prices data to a datetime stamped file."""
        filename = datetime.now(pytz.utc).strftime(
            "prices-%Y-%m-%d.csv"  # TODO: configurable?
        )
        filepath = Path(self.prices_file).parent / filename
        self._last_backup_filename = filepath
        data.to_csv(filepath, index=False)

    def load_prices(self):
        """Loads up and returns the application price data as a DataFrame."""
        if self._prices_data is not None:
            return self._prices_data

        cols = ["author", "kind", "price", "timestamp"]
        dtypes = ["int64", "object", "int64", "datetime64[ns, UTC]"]
        if Path(self.prices_file).exists():
            self._prices_data = pd.read_csv(
                self.prices_file, names=cols, parse_dates=True, skiprows=1
            )
        else:
            self._prices_data = pd.read_csv(
                StringIO(""), names=cols, dtype=dict(zip(cols, dtypes))
            )
        self._prices_data = self._prices_data.astype(dict(zip(cols, dtypes)))
        return self._prices_data

    def save_users(self, data):
        """Saves the given users data to csv file."""
        data.to_csv(self.users_file, index=False)  # persist to disk
        self._users_data = data  # in-memory optimization

    def load_users(self):
        """Returns a DataFrame of user data or creates an empty one."""
        if self._users_data is not None:
            self._users_data = self._users_data.fillna("")
            return self._users_data

        cols = [
            "author",
            "hemisphere",
            "timezone",
            "island",
            "friend",
            "fruit",
            "nickname",
            "creator",
        ]
        dtypes = ["int64", "str", "str", "str", "str", "str", "str", "str"]
        if Path(self.users_file).exists():
            self._users_data = pd.read_csv(self.users_file, names=cols, skiprows=1)
        else:
            self._users_data = pd.read_csv(
                StringIO(""), names=cols, dtype=dict(zip(cols, dtypes))
            )
        self._users_data = self._users_data.fillna("")
        self._users_data = self._users_data.astype(dict(zip(cols, dtypes)))
        return self._users_data

    def save_art(self, data):
        """Saves the given art data to csv file."""
        data.to_csv(self.art_file, index=False)  # persist to disk
        self._art_data = data  # in-memory optimization

    def load_art(self):
        """Returns a DataFrame of art data or creates an empty one."""
        if self._art_data is None:
            try:
                self._art_data = pd.read_csv(self.art_file)
            except FileNotFoundError:
                self._art_data = pd.DataFrame(columns=["author", "name"])
        return self._art_data

    def save_fossils(self, data):
        """Saves the given fossils data to csv file."""
        data.to_csv(self.fossils_file, index=False)  # persist to disk
        self._fossils_data = data  # in-memory optimization

    def load_fossils(self):
        """Returns a DataFrame of fossils data or creates an empty one."""
        if self._fossils_data is None:
            try:
                self._fossils_data = pd.read_csv(self.fossils_file)
            except FileNotFoundError:
                self._fossils_data = pd.DataFrame(columns=["author", "name"])
        return self._fossils_data

    def _get_island_data(self, user):
        timeline = self.get_user_timeline(user.id)
        timeline_data = dict(
            zip(
                [
                    "Sunday_AM",
                    "Monday_AM",
                    "Monday_PM",
                    "Tuesday_AM",
                    "Tuesday_PM",
                    "Wednesday_AM",
                    "Wednesday_PM",
                    "Thursday_AM",
                    "Thursday_PM",
                    "Friday_AM",
                    "Friday_PM",
                    "Saturday_AM",
                    "Saturday_PM",
                ],
                timeline,
            )
        )
        timeline_data = {k: v for k, v in timeline_data.items() if v is not None}
        # TODO: Incorporate information about user's pattern from last week
        return {"initial_week": False, "timeline": timeline_data}

    def _get_predictive_graph(self, target_user, graphname):
        """Builds a predictive graph of a user's price data."""
        islands = {"islands": {}}
        island_data = self._get_island_data(target_user)
        if "Sunday_AM" not in island_data["timeline"]:
            return None
        islands["islands"][target_user.name] = island_data
        arch = Archipelago.load_json(json.dumps(islands))
        island = next(arch.islands)  # there should only be one island
        plot_models_range(
            island.name, list(island.model_group.models), island.previous_week, True
        )

        # fit the y-axis to the plotted data
        maximum = 0
        ax = plt.gca()
        for line in ax.lines:
            for price in line.get_ydata():
                if price > maximum:
                    maximum = price
        for collection in ax.collections:
            for point in collection.get_offsets():
                _, y = point
                if y > maximum:
                    maximum = y
        ax.set_ylim(0, maximum + 50)
        ax.autoscale_view()
        plt.draw()

        plt.savefig(graphname)
        return plt

    def _get_historical_graph(self, channel, graphname):
        """Builds a historical graph of everyone's price data."""
        HOURS = mdates.HourLocator()
        HOURS_FMT = mdates.DateFormatter("%b %d %H:%M")
        TWELVEHOUR = mdates.HourLocator(interval=12)

        plt.figure(figsize=(10, 12), dpi=100)
        _, ax = plt.subplots()
        ax.xaxis.set_major_locator(TWELVEHOUR)
        ax.xaxis.set_major_formatter(HOURS_FMT)
        ax.xaxis.set_minor_locator(HOURS)

        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))

        priceList = self.load_prices()
        legendElems = []

        found_at_least_one_user = False
        for user_id, df in priceList.groupby(by="author"):
            dates = []
            prices = []
            user_name = discord_user_name(channel, user_id)
            if not user_name:
                continue
            found_at_least_one_user = True
            legendElems.append(user_name)
            for _, row in df.iterrows():
                if row.kind == "sell":
                    prices.append(row.price)
                    dates.append(row.timestamp)
            if dates:
                plt.plot(dates, prices, linestyle="-", marker="o", label=user_name)
        if not found_at_least_one_user:
            return None

        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.subplots_adjust(left=0.05, bottom=0.2, right=0.85)
        plt.grid(b=True, which="major", color="#666666", linestyle="-")
        ax.yaxis.grid(b=True, which="minor", color="#555555", linestyle=":")
        plt.ylabel("Price")
        plt.xlabel("Time (UTC)")
        plt.title("Selling Prices")
        plt.legend(legendElems, loc="upper left", bbox_to_anchor=(1, 1))

        figure = plt.gcf()
        figure.set_size_inches(18, 9)

        plt.savefig(graphname, dpi=100)
        return plt

    def get_graph(self, channel, target_user, graphname):
        """Returns a graph of user data; the call site is responsible for closing."""
        if target_user:
            return self._get_predictive_graph(target_user, graphname)
        else:
            return self._get_historical_graph(channel, graphname)

    def generate_graph(self, channel, target_user, graphname):  # pragma: no cover
        """Generates a nice looking graph of user data."""
        fig = self.get_graph(channel, target_user, graphname)
        if fig:
            plt.close("all")

    def append_price(self, author, kind, price, at):
        """Adds a price to the prices data file for the given author and kind."""
        at = datetime.now(pytz.utc) if not at else at
        at = at.astimezone(pytz.utc)  # always store data in UTC
        prices = self.load_prices()
        prices = prices.append(
            pd.DataFrame(columns=prices.columns, data=[[author.id, kind, price, at]]),
            ignore_index=True,
        )
        self.save_prices(prices)

    def get_last_price(self, user_id):
        """Returns the last sell price for the given user id."""
        prices = self.load_prices()
        last = (
            prices[(prices.author == user_id) & (prices.kind == "sell")]
            .sort_values(by=["timestamp"])
            .tail(1)
            .price
        )
        return last.iloc[0] if last.any() else None

    def get_user_prefs(self, user_id):
        users = self.load_users()
        row = users[users.author == user_id].tail(1)
        if row.empty:
            return {}

        prefs = {}
        data = row.to_dict(orient="records")[0]
        for column in users.columns[1:]:
            if not data[column]:
                continue
            if column == "timezone":
                prefs[column] = pytz.timezone(data[column])
            else:
                prefs[column] = data[column]
        return prefs

    def get_user_timeline(self, user_id):
        prices = self.load_prices()
        past = datetime.now(pytz.utc) - timedelta(days=12)
        yours = prices[(prices.author == user_id) & (prices.timestamp > past)]
        yours = yours.sort_values(by=["timestamp"])

        # convert all timestamps to the target user's timezone
        target_timezone = self.get_user_prefs(user_id).get("timezone", pytz.UTC)
        yours["timestamp"] = yours.timestamp.dt.tz_convert(target_timezone)

        recent_buy = yours[yours.kind == "buy"].tail(1)
        if recent_buy.empty:  # no buy found
            return [None] * 13

        buy_date = recent_buy.timestamp.iloc[0]
        if buy_date.to_pydatetime().isoweekday() != 7:  # buy isn't on a sunday
            return [None] * 13

        buy_price = int(recent_buy.price.iloc[0])
        sells = yours[(yours.kind == "sell") & (yours.timestamp > buy_date)]
        if sells.empty:  # no sells found after buy date
            return [buy_price] + [None] * 12

        sell_data = [
            [None, None],  # monday
            [None, None],  # tuesday
            [None, None],  # wednesday
            [None, None],  # friday
            [None, None],  # thrusday
            [None, None],  # saturday
        ]

        groups = sells.set_index("timestamp").groupby(pd.Grouper(freq="D"))
        for day, df in groups:
            day_of_week = day.to_pydatetime().isoweekday()
            if day_of_week == 7:  # no sells allowed on sundays
                continue
            for ts, row in df.iterrows():
                if ts.hour < 12:  # am
                    sell_data[day_of_week - 1][0] = int(row["price"])
                else:  # pm
                    sell_data[day_of_week - 1][1] = int(row["price"])

        timeline = [buy_price]
        for day in sell_data:
            timeline.extend(day)
        return timeline

    def to_usertime(self, author_id, dt):
        user_timezone = self.get_user_prefs(author_id).get("timezone", pytz.UTC)
        if hasattr(dt, "tz_convert"):  # pandas-datetime-like objects
            return dt.tz_convert(user_timezone)
        elif hasattr(dt, "astimezone"):  # python-datetime-like objects
            return dt.astimezone(user_timezone)
        else:  # pragma: no cover
            logging.warning(f"can't convert tz on {dt} for user {author_id}")
            return dt

    def save_user_pref(self, author, pref, value):
        users = self.load_users()
        row = users[users.author == author.id].tail(1)
        if row.empty:
            data = pd.DataFrame(columns=users.columns)
            data["author"] = [author.id]
            data[pref] = [value]
            users = users.append(data, ignore_index=True)
        else:
            users.at[row.index, pref] = value
        self.save_users(users)

    def paginate(self, text):
        """Discord responses must be 2000 characters of less; paginate breaks them up."""
        breakpoints = ["\n", ".", ",", "-"]
        remaining = text
        while len(remaining) > 2000:
            breakpoint = 1999

            for char in breakpoints:
                index = remaining.rfind(char, 1800, 1999)
                if index != -1:
                    breakpoint = index
                    break

            # A trailing blank quoted line just shows the > symbol, so if possible
            # we should try bumping that into the remainder if it is present
            lastnlindex = remaining.rfind("\n", 1800, breakpoint - 1)
            finalline = remaining[lastnlindex:breakpoint]
            if finalline.strip() == ">":
                # the breakpoint should actually be bumped back a bit
                breakpoint = lastnlindex

            yield remaining[0 : breakpoint + 1]
            remaining = remaining[breakpoint + 1 :]

        yield remaining

    async def process(self, message):
        """Process a command message."""
        tokens = message.content.split(" ")
        request, params = tokens[0].lstrip("!").lower(), tokens[1:]
        params = list(filter(None, params))  # ignore any empty string parameters
        if not request:
            return
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        commands = [member[0] for member in members if member[0].endswith("_command")]
        matching = [command for command in commands if command.startswith(request)]
        if not matching:
            await message.channel.send(s("not_a_command", request=request), file=None)
            return
        exact = f"{request}_command"
        if len(matching) > 1 and exact not in matching:
            possible = ", ".join(f"!{m.replace('_command', '')}" for m in matching)
            await message.channel.send(s("did_you_mean", possible=possible), file=None)
        else:
            command = exact if exact in matching else matching[0]
            logging.debug("%s (author=%s, params=%s)", command, message.author, params)
            method = getattr(self, command)
            async with message.channel.typing():
                response, attachment = method(message.channel, message.author, params)
            if not isinstance(response, list):
                response = [response]
            last_reply_index = len(response) - 1
            for n, reply in enumerate(response):
                if isinstance(reply, str):
                    pages = list(self.paginate(reply))
                    last_page_index = len(pages) - 1
                    for i, page in enumerate(pages):
                        file = (
                            attachment
                            if attachment
                            and i == last_page_index
                            and n == last_reply_index
                            else None
                        )
                        await message.channel.send(page, file=file)
                elif isinstance(reply, discord.embeds.Embed):
                    file = attachment if attachment and n == last_reply_index else None
                    await message.channel.send(embed=reply, file=file)

    ##############################
    # Discord Client Behavior
    ##############################

    async def on_message(self, message):
        """Behavior when the client gets a message from Discord."""
        if (
            str(message.channel.type) == "text"
            and message.author.id != self.user.id
            and message.channel.name in self.channels
            and message.content.startswith("!")
        ):
            await self.process(message)

    async def on_ready(self):
        """Behavior when the client has successfully connected to Discord."""
        logging.debug("logged in as %s", self.user)

    ##############################
    # Bot Command Functions
    ##############################

    # Any method of this class with a name that ends in _command is automatically
    # detected as a bot command. These methods should have a signature like:
    #
    #     def your_command(self, channel, author, params)
    #
    # - `channel` is the Discord channel where the command message was sent.
    # - `author` is the Discord author who sent the command.
    # - `params` are any space delimitered parameters also sent with the command.
    #
    # The return value for a command method can be `(string, discord.File)` where the
    # string is the response message the bot should send to the channel and the file
    # object is an attachment to send with the message. For no attachment, use `None`.
    #
    # You can also return `(embed, discord.File)` to respond with an embed (plus the
    # optional attachment as well if so desired).
    #
    # And finally you can also respond with a list if you want the bot to make multiple
    # replies. This works with both embeds and strings. For example:
    #
    #     return ["send", "multiple", "replies"], None
    #
    # Would trigger the bot to send three messages to the channel with no attachment.
    #
    # The docstring used for the command method will be automatically used as the help
    # message for the command. To document commands with parameters use a | to delimit
    # the help message from the parameter documentation. For example:
    #
    #     """This is the help message for your command. | [and] [these] [are] [params]"""
    #
    # A [parameter] is optional whereas a <parameter> is required.

    def help_command(self, channel, author, params):
        """
        Shows this help screen.
        """
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        commands = [member[1] for member in members if member[0].endswith("_command")]
        usage = "__**Turbot Help!**__"
        for command in commands:
            doc = command.__doc__.split("|")
            use, params = doc[0], ", ".join([param.strip() for param in doc[1:]])
            use = inspect.cleandoc(use)
            use = use.replace("\n", " ")

            title = f"!{command.__name__.replace('_command', '')}"
            if params:
                title = f"{title} {params}"
            usage += f"\n> **{title}**"
            usage += f"\n>    {use}"
            usage += "\n> "
        usage += "\n> turbot created by TheAstropath"
        return usage, None

    class _PriceTimeError(Exception):
        def __init__(self, key):
            self.key = key

    def _get_price_time(self, user_id, params):
        if len(params) == 1:
            return None

        if len(params) != 3:
            raise Turbot._PriceTimeError("price_time_invalid")

        day_of_week = params[1].lower()
        if day_of_week not in DAYS:
            raise Turbot._PriceTimeError("day_of_week_invalid")

        time_of_day = params[2].lower()
        if time_of_day not in ["morning", "evening"]:
            raise Turbot._PriceTimeError("time_of_day_invalid")

        now = self.to_usertime(user_id, datetime.now(pytz.utc))
        start = now - timedelta(days=now.isoweekday() % 7)  # start of week
        start = datetime(start.year, start.month, start.day, tzinfo=start.tzinfo)
        day_offset = DAYS[day_of_week]
        hour_offset = 13 if time_of_day == "evening" else 0
        return start + timedelta(days=day_offset, hours=hour_offset)

    def sell_command(self, channel, author, params):
        """
        Log the price that you can sell turnips for on your island.
        | <price> [day time]
        """
        if not params:
            return s("sell_no_params"), None

        price = params[0]
        if not price.isnumeric():
            return s("sell_nonnumeric_price"), None

        price = int(price)
        if price <= 0:
            return s("sell_nonpositive_price"), None

        last_price = self.get_last_price(author.id)

        try:
            price_time = self._get_price_time(author.id, params)
        except Turbot._PriceTimeError as err:
            return s(err.key), None

        logging.debug("saving sell price of %s bells for user id %s", price, author.id)
        self.append_price(author=author, kind="sell", price=price, at=price_time)

        key = (
            "sell_new_price"
            if not last_price or price_time is not None
            else "sell_higher_price"
            if price > last_price
            else "sell_lower_price"
            if price < last_price
            else "sell_same_price"
        )
        return s(key, price=price, name=author, last_price=last_price), None

    def buy_command(self, channel, author, params):
        """
        Log the price that you can buy turnips from Daisy Mae on your island.
        | <price> [day time]
        """
        if not params:
            return s("buy_no_params"), None

        price = params[0]
        if not price.isnumeric():
            return s("buy_nonnumeric_price"), None

        price = int(price)
        if price <= 0:
            return s("buy_nonpositive_price"), None

        try:
            price_time = self._get_price_time(author.id, params)
        except Turbot._PriceTimeError as err:
            return s(err.key), None

        logging.debug("saving buy price of %s bells for user id %s", price, author.id)
        self.append_price(author=author, kind="buy", price=price, at=price_time)

        return s("buy", price=price, name=author), None

    def reset_command(self, channel, author, params):
        """
        Only Turbot Admin members can run this command. Generates a final graph for use
        with !lastweek and resets all data for all users.
        """
        if not is_turbot_admin(channel, author):
            return s("not_admin"), None

        self.generate_graph(channel, None, LASTWEEKCMD_FILE)
        prices = self.load_prices()
        self.backup_prices(prices)

        buys = prices[prices.kind == "buy"].sort_values(by="timestamp")
        idx = buys.groupby(by="author")["timestamp"].idxmax()
        prices = buys.loc[idx]
        self.save_prices(prices)
        return s("reset"), None

    def lastweek_command(self, channel, author, params):
        """
        Displays the final graph from the last week before the data was reset.
        """
        if not Path(LASTWEEKCMD_FILE).exists():
            return s("lastweek_none"), None
        return s("lastweek"), discord.File(LASTWEEKCMD_FILE)

    def graph_command(self, channel, author, params):
        """
        Generates a graph of turnip prices for all users. If a user is specified, only
        graph that users prices. | [user]
        """

        if not params:
            self.generate_graph(channel, None, GRAPHCMD_FILE)
            return s("graph_all_users"), discord.File(GRAPHCMD_FILE)

        user_id = discord_user_id(channel, params[0])
        user_name = discord_user_name(channel, user_id)
        user = discord_user_from_name(channel, user_name)
        if not user:
            return s("cant_find_user", name=params[0]), None

        self.generate_graph(channel, user, GRAPHCMD_FILE)
        return s("graph_user", name=user_name), discord.File(GRAPHCMD_FILE)

    def turnippattern_command(self, channel, author, params):
        """
        Calculates the patterns you will see in your shop based on Daisy Mae's price
        on your island and your Monday morning sell price. |
        <Sunday Buy Price> <Monday Morning Sell Price>
        """
        if len(params) != 2:
            return s("turnippattern_bad_params"), None

        buyprice, mondayprice = params
        if not buyprice.isnumeric() or not mondayprice.isnumeric():
            return s("turnippattern_nonnumeric_price"), None

        buyprice, mondayprice = int(buyprice), int(mondayprice)
        xval = mondayprice / buyprice
        patterns = (
            [1, 4]
            if xval >= 0.91
            else [2, 3, 4]
            if xval >= 0.85
            else [3, 4]
            if xval >= 0.80
            else [1, 4]
            if xval >= 0.60
            else [4]
        )
        lines = [s("turnippattern_header")]
        if 1 in patterns:
            lines.append(s("turnippattern_pattern1"))
        if 2 in patterns:
            lines.append(s("turnippattern_pattern2"))
        if 3 in patterns:
            lines.append(s("turnippattern_pattern3"))
        lines.append(s("turnippattern_pattern4"))  # pattern 4 is always possible
        return "\n".join(lines), None

    def history_command(self, channel, author, params):
        """
        Show the historical turnip prices for a user. If no user is specified, it will
        display your own prices. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        prices = self.load_prices()
        yours = prices[prices.author == target_id]
        lines = [s("history_header", name=target_name)]
        for _, row in yours.iterrows():
            lines.append(
                s(
                    f"history_{row.kind}",
                    price=row.price,
                    timestamp=h(self.to_usertime(target_id, row.timestamp)),
                )
            )
        return "\n".join(lines), None

    def oops_command(self, channel, author, params):
        """
        Remove your last logged turnip price.
        """
        target = author.id
        target_name = discord_user_name(channel, target)
        prices = self.load_prices()
        prices = prices.drop(prices[prices.author == author.id].tail(1).index)
        self.save_prices(prices)
        return s("oops", name=target_name), None

    def clear_command(self, channel, author, params):
        """
        Clears all of your own historical turnip prices.
        """
        user_id = discord_user_id(channel, str(author))
        prices = self.load_prices()
        prices = prices[prices.author != user_id]
        self.save_prices(prices)
        return s("clear", name=author), None

    def _best(self, channel, author, kind):
        prices = self.load_prices()
        past = datetime.now(pytz.utc) - timedelta(hours=12)
        sells = prices[(prices.kind == kind) & (prices.timestamp > past)]
        idx = sells.groupby(by="author").price.transform(max) == sells.price
        bests = sells[idx].sort_values(by="price", ascending=kind == "buy")
        lines = [s(f"best{kind}_header")]
        for _, row in bests.iterrows():
            name = discord_user_from_id(channel, row.author)
            lines.append(
                s(
                    "best",
                    name=name,
                    price=row.price,
                    timestamp=h(self.to_usertime(row.author, row.timestamp)),
                )
            )
        return "\n".join(lines), None

    def bestbuy_command(self, channel, author, params):
        """
        Finds the best (and most recent) buying prices logged in the last 12 hours.
        """
        return self._best(channel, author, "buy")

    def bestsell_command(self, channel, author, params):
        """
        Finds the best (and most recent) selling prices logged in the last 12 hours.
        """
        return self._best(channel, author, "sell")

    def collect_command(self, channel, author, params):
        """
        Mark collectables as donated to your museum. The names must match the in-game item
        name exactly. | <comma, separated, list, of, things>
        """
        if not params:
            return s("collect_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))

        valid_fossils = items.intersection(FOSSILS_SET)
        valid_bugs = items.intersection(BUGS_SET)
        valid_fish = items.intersection(FISH_SET)
        valid_art = items.intersection(ART_SET)
        invalid = items.difference(COLLECTABLE_SET)

        lines = []

        if valid_fossils:
            fossils = self.load_fossils()
            yours = fossils[fossils.author == author.id]
            dupes = yours.loc[yours.name.isin(valid_fossils)].name.values.tolist()
            new_names = list(set(valid_fossils) - set(dupes))
            new_data = [[author.id, name] for name in new_names]
            new_fossils = pd.DataFrame(columns=fossils.columns, data=new_data)
            fossils = fossils.append(new_fossils, ignore_index=True)
            yours = fossils[fossils.author == author.id]  # re-fetch for congrats
            self.save_fossils(fossils)
            if new_names:
                lines.append(s("collect_fossil_new", items=", ".join(sorted(new_names))))
            if dupes:
                lines.append(s("collect_fossil_dupe", items=", ".join(sorted(dupes))))
            if len(FOSSILS_SET) == len(yours.index):
                lines.append(s("congrats_all_fossils"))

        if valid_bugs:
            lines.append(s("collect_bugs"))

        if valid_fish:
            lines.append(s("collect_fish"))

        if valid_art:
            art = self.load_art()
            yours = art[art.author == author.id]
            dupes = yours.loc[yours.name.isin(valid_art)].name.values.tolist()
            new_names = list(set(valid_art) - set(dupes))
            new_data = [[author.id, name] for name in new_names]
            new_art = pd.DataFrame(columns=art.columns, data=new_data)
            art = art.append(new_art, ignore_index=True)
            yours = art[art.author == author.id]  # re-fetch for congrats
            self.save_art(art)
            if new_names:
                lines.append(s("collect_art_new", items=", ".join(sorted(new_names))))
            if dupes:
                lines.append(s("collect_art_dupe", items=", ".join(sorted(dupes))))
            if len(ART) == len(yours.index):
                lines.append(s("congrats_all_art"))

        if invalid:
            lines.append(s("invalid_collectable", items=", ".join(sorted(invalid))))

        return "\n".join(lines), None

    def uncollect_command(self, channel, author, params):
        """
        Unmark collectables as donated to your museum. The names must match the in-game
        item name exactly. | <comma, separated, list, of, things>
        """
        if not params:
            return s("uncollect_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))

        valid_fossils = items.intersection(FOSSILS_SET)
        valid_bugs = items.intersection(BUGS_SET)
        valid_fish = items.intersection(FISH_SET)
        valid_art = items.intersection(ART_SET)
        invalid = items.difference(COLLECTABLE_SET)

        lines = []

        if valid_fossils:
            fossils = self.load_fossils()
            yours = fossils[fossils.author == author.id]
            previously_collected = yours.loc[yours.name.isin(valid_fossils)]
            deleted = set(previously_collected.name.values.tolist())
            didnt_have = valid_fossils - deleted
            fossils = fossils.drop(previously_collected.index)
            self.save_fossils(fossils)
            if deleted:
                lines.append(
                    s("uncollect_fossil_deleted", items=", ".join(sorted(deleted)))
                )
            if didnt_have:
                lines.append(
                    s("uncollect_fossil_already", items=", ".join(sorted(didnt_have)))
                )

        if valid_bugs:
            lines.append(s("uncollect_bugs"))

        if valid_fish:
            lines.append(s("uncollect_fish"))

        if valid_art:
            art = self.load_art()
            yours = art[art.author == author.id]
            previously_collected = yours.loc[yours.name.isin(valid_art)]
            deleted = set(previously_collected.name.values.tolist())
            didnt_have = valid_art - deleted
            art = art.drop(previously_collected.index)
            self.save_art(art)
            if deleted:
                lines.append(s("uncollect_art_deleted", items=", ".join(sorted(deleted))))
            if didnt_have:
                lines.append(
                    s("uncollect_art_already", items=", ".join(sorted(didnt_have)))
                )

        if invalid:
            lines.append(s("invalid_collectable", items=", ".join(sorted(invalid))))

        return "\n".join(lines), None

    def search_command(self, channel, author, params):
        """
        Searches all users to see who needs the given collectables. The names must match
        the in-game item name, and more than one can be provided if separated by commas.
        | <list of collectables>
        """
        if not params:
            return s("search_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))

        valid_fossils = items.intersection(FOSSILS_SET)
        valid_bugs = items.intersection(BUGS_SET)
        valid_fish = items.intersection(FISH_SET)
        valid_art = items.intersection(ART_SET)
        invalid = items.difference(COLLECTABLE_SET)

        fossils = self.load_fossils()
        fossil_users = fossils.author.unique()
        fossil_results = defaultdict(list)
        for fossil in valid_fossils:
            havers = fossils[fossils.name == fossil].author.unique()
            needers = np.setdiff1d(fossil_users, havers).tolist()
            for needer in needers:
                name = discord_user_from_id(channel, needer)
                fossil_results[name].append(fossil)

        if valid_bugs:
            return s("search_bugs"), None

        if valid_fish:
            return s("search_fish"), None

        art = self.load_art()
        art_users = art.author.unique()
        art_results = defaultdict(list)
        for artpiece in valid_art:
            havers = art[art.name == artpiece].author.unique()
            needers = np.setdiff1d(art_users, havers).tolist()
            for needer in needers:
                name = discord_user_from_id(channel, needer)
                art_results[name].append(artpiece)

        if not fossil_results and not art_results and not invalid:
            return s("search_all_not_needed"), None

        searched = valid_fossils | valid_bugs | valid_fish | valid_art
        needed = set()
        for items in fossil_results.values():
            needed.update(items)
        for items in art_results.values():
            needed.update(items)
        not_needed = searched - needed

        lines = []
        for name, items in fossil_results.items():
            items_str = ", ".join(sorted(items))
            lines.append(s("search_fossil_row", name=name, items=items_str))
        for name, items in art_results.items():
            items_str = ", ".join(sorted(items))
            lines.append(s("search_art_row", name=name, items=items_str))
        if not_needed:
            items_str = ", ".join(sorted(not_needed))
            lines.append(s("search_not_needed", items=items_str))
        if invalid:
            lines.append(s("search_invalid", items=", ".join(sorted(invalid))))
        return "\n".join(sorted(lines)), None

    def allfossils_command(self, channel, author, params):
        """
        Shows all possible fossils that you can donate to the museum.
        """
        return s("allfossils", list=", ".join(sorted(FOSSILS_SET))), None

    def uncollected_command(self, channel, author, params):
        """
        Lists all collectables that you still need to donate. If a user is provided, it
        gives the same information for that user instead. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        fossils = self.load_fossils()
        your_fossils = fossils[fossils.author == target_id]
        collected_fossils = set(your_fossils.name.unique())
        remaining_fossils = FOSSILS_SET - collected_fossils

        art = self.load_art()
        your_art = art[art.author == target_id]
        collected_art = set(your_art.name.unique())
        remaining_art = ART_SET - collected_art

        lines = []

        if remaining_fossils:
            lines.append(
                s(
                    "uncollected_fossils_count",
                    count=len(remaining_fossils),
                    name=target_name,
                )
            )
            lines.append(
                s(
                    "uncollected_fossils_remaining",
                    items=", ".join(sorted(remaining_fossils)),
                )
            )
        else:
            lines.append(s("congrats_all_fossils"))

        if remaining_art:
            lines.append(
                s("uncollected_art_count", count=len(remaining_art), name=target_name)
            )
            lines.append(
                s("uncollected_art_remaining", items=", ".join(sorted(remaining_art)))
            )
        else:
            lines.append(s("congrats_all_art"))

        return "\n".join(lines), None

    def neededfossils_command(self, channel, author, params):
        """
        Lists all the needed fossils for all the channel members.
        """
        fossils = self.load_fossils()
        authors = [member.id for member in channel.members if member.id != self.user.id]
        total = pd.DataFrame(
            list(product(authors, FOSSILS_SET)), columns=["author", "name"]
        )
        merged = total.merge(fossils, indicator=True, how="outer")
        needed = merged[merged["_merge"] == "left_only"]

        lines = []
        for user, df in needed.groupby(by="author"):
            name = discord_user_name(channel, user)
            items_list = sorted([row["name"] for _, row in df.iterrows()])
            if len(items_list) == len(FOSSILS_SET):
                continue
            elif len(items_list) > 10:
                items_str = "_more than 10 fossils..._"
            else:
                items_str = ", ".join(items_list)
            lines.append(s("neededfossils", name=name, items=items_str))
        if not lines:
            return s("neededfossils_none"), None
        return "\n".join(sorted(lines)), None

    def collected_command(self, channel, author, params):
        """
        Lists all collectables that you have already donated. If a user is provided, it
        gives the same information for that user instead. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        fossils = self.load_fossils()
        your_fossils = fossils[fossils.author == target_id]
        collected_fossils = set(your_fossils.name.unique())
        all_fossils = len(collected_fossils) == len(FOSSILS_SET)

        art = self.load_art()
        your_art = art[art.author == target_id]
        collected_art = set(your_art.name.unique())
        all_art = len(collected_art) == len(ART_SET)

        lines = []
        if any([all_fossils, all_art]):
            if all_fossils:
                lines.append(s("congrats_all_fossils"))
            if all_art:
                lines.append(s("congrats_all_art"))
            if all([all_fossils, all_art]):
                return "\n".join(lines), None

        if collected_art and not all_art:
            lines.append(
                s(
                    "collected_art",
                    name=target_name,
                    count=len(collected_art),
                    items=", ".join(sorted(collected_art)),
                )
            )
        if collected_fossils and not all_fossils:
            lines.append(
                s(
                    "collected_fossils",
                    name=target_name,
                    count=len(collected_fossils),
                    items=", ".join(sorted(collected_fossils)),
                )
            )
        return "\n".join(lines), None

    def predict_command(self, channel, author, params):
        """
        Get a link to a prediction calulator for a price history. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        timeline = self.get_user_timeline(target_id)
        if not timeline[0]:
            return s("cant_find_buy", name=target_name), None

        query = ".".join((str(price) if price else "") for price in timeline).rstrip(".")
        url = f"{self.base_prophet_url}{query}"
        return s("predict", name=target_name, url=url), None

    def friend_command(self, channel, author, params):
        """
        Set your friend code. | <code>
        """
        if not params:
            return s("friend_no_params"), None

        code = re.sub("[^0-9]", "", "".join(params).replace("-", ""))
        if len(code) != 12 or not code.isdigit():
            return s("friend_invalid"), None

        self.save_user_pref(author, "friend", code)
        return s("friend", name=author), None

    def creator_command(self, channel, author, params):
        """
        Set your creator code. | <code>
        """
        if not params:
            return s("creator_no_params"), None

        code = re.sub("[^0-9]", "", "".join(params).replace("-", ""))
        if len(code) != 12 or not code.isdigit():
            return s("creator_invalid"), None

        self.save_user_pref(author, "creator", code)
        return s("creator", name=author), None

    def fruit_command(self, channel, author, params):
        """
        Set your island's native fruit. | [apple|cherry|orange|peach|pear]
        """
        if not params:
            return s("fruit_no_params"), None

        fruit = params[0].lower()
        if fruit not in ["apple", "cherry", "orange", "peach", "pear"]:
            return s("fruit_invalid"), None

        self.save_user_pref(author, "fruit", fruit)
        return s("fruit", name=author), None

    def hemisphere_command(self, channel, author, params):
        """
        Set your hemisphere. | [Northern|Southern]
        """
        if not params:
            return s("hemisphere_no_params"), None

        home = params[0].lower()
        if home not in ["northern", "southern"]:
            return s("hemisphere_bad_params"), None

        self.save_user_pref(author, "hemisphere", home)
        return s("hemisphere", name=author), None

    def nickname_command(self, channel, author, params):
        """
        Set your nickname, such as your Switch user name. | <name>
        """
        if not params:
            return s("nickname_no_params"), None

        name = " ".join(params)  # allow spaces in nicknames
        self.save_user_pref(author, "nickname", name)
        return s("nickname", name=author), None

    def timezone_command(self, channel, author, params):
        """
        Set your timezone. You can find a list of supported TZ names at
        <https://en.wikipedia.org/wiki/List_of_tz_database_time_zones> | <zone>
        """
        if not params:
            return s("timezone_no_params"), None

        zone = params[0]
        if zone not in pytz.all_timezones_set:
            return s("timezone_bad_params"), None

        self.save_user_pref(author, "timezone", zone)
        return s("timezone", name=author), None

    def island_command(self, channel, author, params):
        """
        Set your island name. | <name>
        """
        if not params:
            return s("island_no_params"), None

        island = " ".join(params)  # allow spaces in island names
        self.save_user_pref(author, "island", island)
        return s("island", name=author), None

    def count_command(self, channel, author, params):
        """
        Provides a count of the number of pieces of collectables for the comma-separated
        list of users. | <list of users>
        """
        if not params:
            return s("count_no_params"), None

        users = set(item.strip().lower() for item in " ".join(params).split(","))

        valid = []
        invalid = []
        for user in users:
            user_name = discord_user_name(channel, user)
            user_id = discord_user_id(channel, user_name)
            if user_name and user_id:
                valid.append((user_name, user_id))
            else:
                invalid.append(user)

        lines = []
        if valid:
            lines.append(s("count_fossil_valid_header"))
            fossils = self.load_fossils()
            for user_name, user_id in sorted(valid):
                yours = fossils[fossils.author == user_id]
                collected = set(yours.name.unique())
                remaining = FOSSILS_SET - collected
                lines.append(
                    s("count_fossil_valid", name=user_name, count=len(remaining))
                )

            lines.append(s("count_art_valid_header"))
            art = self.load_art()
            for user_name, user_id in sorted(valid):
                yours = art[art.author == user_id]
                collected = set(yours.name.unique())
                allnames = set(ART.name.unique())
                remaining = allnames - collected
                lines.append(s("count_art_valid", name=user_name, count=len(remaining)))

        if invalid:
            lines.append(s("count_invalid_header"))
            for user in invalid:
                lines.append(s("count_invalid", name=user))

        return "\n".join(lines), None

    def art_command(self, channel, author, params):
        """
        Get info about pieces of art that are available | [List of art pieces]
        """
        response = ""
        if params:
            items = set(item.strip().lower() for item in " ".join(params).split(","))
            validset = items.intersection(ART_SET)
            invalidset = items - validset
            valid = sorted(list(validset))
            invalid = sorted(list(invalidset))
            lines = []
            response = s("art_header") + "\n"
            for art in valid:
                piece = ART[ART.name == art].iloc[0]
                if piece["has_fake"]:
                    lines.append(
                        s(
                            "art_fake",
                            name=piece["name"].title(),
                            desc=piece["fake_description"],
                            real_url=piece["real_image_url"],
                            fake_url=piece["fake_image_url"],
                        )
                    )
                else:
                    lines.append(
                        s(
                            "art_real",
                            name=piece["name"].title(),
                            real_url=piece["real_image_url"],
                        )
                    )

            response += "\n> \n".join(lines)

            if invalid:
                response += "\n" + (s("art_invalid", items=", ".join(invalid)))

        else:
            response = s("allart", list=", ".join(sorted(ART_SET)))

        return response, None

    def _creatures(self, *_, author, params, kind, source, force_text=False):
        """The fish and bugs commands are so similar; I factored them out to a helper."""
        hemisphere = self.get_user_prefs(author.id).get("hemisphere", None)
        if not hemisphere:
            return s("no_hemisphere")

        now = self.to_usertime(author.id, datetime.now(pytz.utc))
        this_month = now.strftime("%b").lower()
        first_this_month = now.replace(day=1)
        last_month = (first_this_month - timedelta(days=1)).strftime("%b").lower()
        next_month = (first_this_month + relativedelta(months=1)).strftime("%b").lower()
        available = source[(source.hemisphere == hemisphere) & (source[this_month] == 1)]

        def details(row):
            alert = (
                "**GONE NEXT MONTH!**"
                if not row[next_month]
                else "_New this month_"
                if not row[last_month]
                else ""
            )
            months = ", ".join(list(humanize_months(row)))
            return {
                **row,
                "name": row["name"].capitalize(),
                "months": months,
                "alert": alert,
            }

        def add_header(lines):
            if kind != "bugs":
                return lines
            if random.randint(0, 100) > 70:
                lines.insert(0, s("bugs_header"))
            return lines

        if params:
            user_input = " ".join(params)
            search = user_input.lower()
            if search == "leaving":
                found = available[available[next_month] == 0]
            elif search == "arriving":
                found = available[available[last_month] == 0]
            else:
                found = available[available.name.str.contains(search)]

            if found.empty:
                return s(f"{kind}_none_found", search=user_input)
            elif force_text or len(found) > EMBED_LIMIT:
                available = found  # fallback to the less detailed, text only, response
            else:
                response = []
                rows = [row for _, row in found.iterrows()]
                for row in sorted(rows, key=lambda r: r["name"]):
                    info = details(row)
                    embed = discord.Embed(title=info["name"])
                    embed.set_thumbnail(url=info["image"])
                    embed.add_field(name="price", value=info["price"])
                    embed.add_field(name="location", value=info["location"])
                    if "shadow" in info:
                        embed.add_field(name="shadow size", value=info["shadow"])
                    embed.add_field(name="available", value=info["time"])
                    embed.add_field(name="during", value=info["months"])
                    if info["alert"]:
                        embed.add_field(name="alert", value=info["alert"])
                        if "GONE" in info["alert"]:
                            embed.color = discord.Color.orange()
                        else:
                            embed.color = discord.Color.blue()
                    response.append(embed)
                return add_header(response)

        lines = [s(kind, **details(row)) for _, row in available.iterrows()]
        return "\n".join(add_header(sorted(lines)))

    def fish_command(self, channel, author, params):
        """
        Tells you what fish are available now in your hemisphere.
        | [name|leaving|arriving]
        """
        return (
            self._creatures(author=author, params=params, kind="fish", source=FISH),
            None,
        )

    def bugs_command(self, channel, author, params):
        """
        Tells you what bugs are available now in your hemisphere.
        | [name|leaving|arriving]
        """
        return (
            self._creatures(author=author, params=params, kind="bugs", source=BUGS),
            None,
        )

    def new_command(self, channel, author, params):
        """
        Tells you what new things available in your hemisphere right now.
        """
        return (
            [
                self._creatures(
                    author=author,
                    params=["arriving"],
                    kind="bugs",
                    source=BUGS,
                    force_text=True,
                ),
                self._creatures(
                    author=author,
                    params=["arriving"],
                    kind="fish",
                    source=FISH,
                    force_text=True,
                ),
            ],
            None,
        )

    def _info_embed(self, user):
        prefs = self.get_user_prefs(user.id)

        embed = discord.Embed(title=user.name)
        embed.set_thumbnail(url=user.avatar_url)

        nickname = prefs.get("nickname", None)
        if nickname:
            embed.add_field(name="Nickname", value=nickname)
        else:
            embed.add_field(name="Nickname", value="Not set")

        code = prefs.get("friend", None)
        if code:
            code_str = f"SW-{code[0:4]}-{code[4:8]}-{code[8:12]}"
            embed.add_field(name="Friend code", value=code_str)
        else:
            embed.add_field(name="Friend code", value="Not set")

        code = prefs.get("creator", None)
        if code:
            code_str = f"MA-{code[0:4]}-{code[4:8]}-{code[8:12]}"
            embed.add_field(name="Creator code", value=code_str)
        else:
            embed.add_field(name="Creator code", value="Not set")

        island = prefs.get("island", "Not set")
        embed.add_field(name="Island", value=island)

        hemisphere = prefs.get("hemisphere", "Not set").title()
        embed.add_field(name="Hemisphere", value=hemisphere)

        fruit = prefs.get("fruit", "Not set").title()
        embed.add_field(name="Native fruit", value=fruit)

        now = self.to_usertime(user.id, datetime.now(pytz.UTC))
        current_time = now.strftime("%I:%M %p %Z")
        embed.set_footer(text=f"Current time is {current_time}")

        return embed

    def info_command(self, channel, author, params):
        """
        Gives you information on a user. | [user]
        """
        if len(params) < 1:
            return s("info_no_params"), None

        query = " ".join(params).lower()  # allow spaces in names

        users = self.load_users()
        for _, row in users.iterrows():
            user_id = int(row["author"])
            user_name = discord_user_name(channel, user_id)
            if not user_name:
                continue
            if user_name.lower().find(query) != -1:
                user = discord_user_from_name(channel, user_name)
                return self._info_embed(user), None

        # check if user exists, they just don't have any info
        for member in channel.members:
            if member.name.lower().find(query) != -1:
                return s("info_no_prefs", user=member), None

        return s("info_not_found"), None

    def about_command(self, channel, author, params):
        """
        Get information about Turbot.
        """
        embed = discord.Embed(title="Turbot")
        embed.set_thumbnail(
            url="https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png"
        )
        embed.add_field(
            name="Version",
            value=f"[{__version__}](https://pypi.org/project/turbot/{__version__}/)",
        )
        embed.add_field(name="Package", value="[PyPI](https://pypi.org/project/turbot/)")
        embed.add_field(
            name="Author", value="[TheAstropath](https://github.com/theastropath)"
        )
        embed.add_field(
            name="Maintainer", value="[lexicalunit](https://github.com/lexicaluit)"
        )
        embed.description = (
            "A Discord bot for everything _Animal Crossing: New Horizons._\n"
            "\n"
            "Use the command `!help` for usage details. Having issues with Turbot? "
            "Please [report bugs](https://github.com/theastropath/turbot/issues)!\n"
        )
        embed.url = "https://github.com/theastropath/turbot"
        embed.set_footer(text="MIT © TheAstropath, lexicalunit et al")
        embed.color = discord.Color(0xFFFDC3)
        return embed, None


def get_token(token_file):  # pragma: no cover
    """Returns the discord token from the environment or your token config file."""
    token = getenv("TURBOT_TOKEN", None)
    if token:
        return token

    try:
        with open(token_file, "r") as f:
            return f.readline().strip()
    except IOError as e:
        with redirect_stdout(sys.stderr):
            print("error:", e)
            print(f"put your discord token in a file named '{token_file}'")
        sys.exit(1)


def get_channels(channels_file):  # pragma: no cover
    """Returns the authorized channels your channels config file."""
    try:
        with open(channels_file, "r") as channels_file:
            return [line.strip() for line in channels_file.readlines()]
    except IOError:
        return []


@click.command()
@click.option(
    "-l",
    "--log-level",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
    default="ERROR",
)
@click.option("-v", "--verbose", count=True, help="sets log level to DEBUG")
@click.option(
    "-b",
    "--bot-token-file",
    default=DEFAULT_CONFIG_TOKEN,
    help="read your discord bot token from this file; "
    "you can also set the token directly via the environment variable TURBOT_TOKEN",
)
@click.option(
    "-c",
    "--channel",
    multiple=True,
    help="authorize a channel; use this multiple times to authorize multiple channels",
)
@click.option(
    "-a",
    "--auth-channels-file",
    default=DEFAULT_CONFIG_CHANNELS,
    help="read authorized channel names from this file",
)
@click.option(
    "-p",
    "--prices-file",
    default=DEFAULT_DB_PRICES,
    help="read price data from this file",
)
@click.option(
    "-a", "--art-file", default=DEFAULT_DB_ART, help="read art data from this file",
)
@click.option(
    "-f",
    "--fossils-file",
    default=DEFAULT_DB_FOSSILS,
    help="read fossil data from this file",
)
@click.option(
    "-u",
    "--users-file",
    default=DEFAULT_DB_USERS,
    help="read users preferences data from this file",
)
@click.version_option(version=__version__)
@click.option(
    "--dev",
    default=False,
    is_flag=True,
    help="Development mode, automatically reload bot when source changes",
)
def main(
    log_level,
    verbose,
    bot_token_file,
    channel,
    auth_channels_file,
    prices_file,
    art_file,
    fossils_file,
    users_file,
    dev,
):  # pragma: no cover
    auth_channels = get_channels(auth_channels_file) + list(channel)
    if not auth_channels:
        print("error: you must provide at least one authorized channel", file=sys.stderr)
        sys.exit(1)

    if dev:
        reloader = hupper.start_reloader("turbot.main")
        reloader.watch_files([ART_DATA_FILE, BUGS_DATA_FILE, FISH_DATA_FILE])

    # ensure transient application directories exist
    DB_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    Turbot(
        token=get_token(bot_token_file),
        channels=auth_channels,
        prices_file=prices_file,
        art_file=art_file,
        fossils_file=fossils_file,
        users_file=users_file,
        log_level=getattr(logging, "DEBUG" if verbose else log_level),
    ).run()


if __name__ == "__main__":
    main()
