import inspect
import json
import logging
import random
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from itertools import product
from os import getenv
from pathlib import Path
from subprocess import run

import click
import discord
import hupper
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from humanize import naturaltime
from unidecode import unidecode

from turbot._version import __version__
from turbot.assets import Assets, s
from turbot.data import Data
from turnips.archipelago import Archipelago
from turnips.plots import plot_models_range

matplotlib.use("Agg")

RUNTIME_ROOT = Path(".")
SCRIPTS_DIR = RUNTIME_ROOT / "scripts"
DEFAULT_DB_DIR = RUNTIME_ROOT / "db"
DEFAULT_CONFIG_TOKEN = RUNTIME_ROOT / "token.txt"
DEFAULT_CONFIG_CHANNELS = RUNTIME_ROOT / "channels.txt"
TMP_DIR = RUNTIME_ROOT / "tmp"
GRAPHCMD_FILE = TMP_DIR / "graphcmd.png"
LASTWEEKCMD_FILE = TMP_DIR / "lastweek.png"
MIGRATIONS_FILE = DEFAULT_DB_DIR / "migrations.txt"
MIGRATIONS_DIR = SCRIPTS_DIR / "migrations"

EMBED_LIMIT = 5  # more embeds in a row than this causes issues

USER_PREFRENCES = [
    "hemisphere",
    "timezone",
    "island",
    "friend",
    "fruit",
    "nickname",
    "creator",
]

# Based on values from datetime.isoweekday()
DAYS = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}
IDAYS = dict(map(reversed, DAYS.items()))


class Validate:
    FRUITS = ["apple", "cherry", "orange", "peach", "pear"]
    HEMISPHERES = ["northern", "southern"]

    @classmethod
    def friend(cls, value):
        code = re.sub("[^0-9]", "", value)
        return code if len(code) == 12 and code.isdigit() else None

    @classmethod
    def creator(cls, value):
        code = re.sub("[^0-9]", "", value)
        return code if len(code) == 12 and code.isdigit() else None

    @classmethod
    def fruit(cls, value):
        fruit = value.lower()
        return fruit if fruit in cls.FRUITS else None

    @classmethod
    def hemisphere(cls, value):
        home = value.lower()
        return home if home in cls.HEMISPHERES else None

    @classmethod
    def nickname(cls, value):
        return value

    @classmethod
    def timezone(cls, value):
        return value if value in pytz.all_timezones_set else None

    @classmethod
    def island(cls, value):
        return value


def h(dt):
    """Convertes a datetime to something readable by a human."""
    if hasattr(dt, "tz_convert"):  # pandas-datetime-like objects
        dt = dt.to_pydatetime()
    naive_dt = dt.replace(tzinfo=None)
    return naturaltime(naive_dt)


def day_and_time(dt):
    """Converts a datetime to a day and time of day, eg: Monday pm."""
    day = IDAYS[dt.isoweekday()]
    am_pm = "am" if dt.hour < 12 else "pm"
    return f"{day.title()} {am_pm}"


def sanitize_collectable(item):
    return unidecode(item.strip().lower())


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
    if user_id is None:
        return None
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


def command(f):
    f.is_command = True
    return f


class Turbot(discord.Client):
    """Discord turnip bot"""

    def __init__(
        self, token="", channels=[], db_dir=DEFAULT_DB_DIR, log_level=None,
    ):
        if log_level:  # pragma: no cover
            logging.basicConfig(level=log_level)
        super().__init__()
        self.token = token
        self.channels = channels
        self.base_prophet_url = "https://turnipprophet.io/?prices="
        self._last_backup_filename = None
        self.data = Data(db_dir=db_dir)

        # build a list of commands supported by this bot by fetching @command methods
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        self._commands = [
            member[0]
            for member in members
            if hasattr(member[1], "is_command") and member[1].is_command
        ]

        # load static application data
        self.assets = Assets()

    def run(self):  # pragma: no cover
        super().run(self.token)

    def last_backup_filename(self):
        """Return the name of the last known backup file for prices or None if unknown."""
        return self._last_backup_filename

    def backup_prices(self, data):
        """Backs up the prices data to a datetime stamped file."""
        filename = datetime.now(pytz.utc).strftime(
            "prices-%Y-%m-%d.csv"  # TODO: configurable?
        )
        filepath = Path(self.data.file("prices")).parent / filename
        self._last_backup_filename = filepath
        data.to_csv(filepath, index=False)

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

        ax = plt.gca()
        if "!!ERROR!!" in ax.get_title():
            return None

        # fit the y-axis to the plotted data
        maximum = 0
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

        priceList = self.data.prices
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
        plot = self.get_graph(channel, target_user, graphname)
        success = plot is not None
        plt.close("all")
        return success

    def append_price(self, author, kind, price, at):
        """Adds a price to the prices data file for the given author and kind."""
        at = datetime.now(pytz.utc) if not at else at
        at = at.astimezone(pytz.utc)  # always store data in UTC
        prices = self.data.prices
        row = pd.DataFrame(columns=prices.columns, data=[[author.id, kind, price, at]])
        prices = prices.append(row, ignore_index=True)
        self.data.commit(prices)

    def get_last_price(self, user_id):
        """Returns the last sell price for the given user id."""
        prices = self.data.prices
        last = (
            prices[(prices.author == user_id) & (prices.kind == "sell")]
            .sort_values(by=["timestamp"])
            .tail(1)
            .price
        )
        return last.iloc[0] if last.any() else None

    def get_user_prefs(self, user_id):
        users = self.data.users
        row = users[users.author == user_id].tail(1)
        if row.empty:
            return {}

        prefs = {}
        data = row.to_dict(orient="records")[0]
        for column in users.columns[1:]:
            if not data[column]:
                continue
            if column == "timezone":
                butt = data[column]
                prefs[column] = pytz.timezone(butt)
            else:
                prefs[column] = data[column]
        return prefs

    def get_user_timeline(self, user_id):
        prices = self.data.prices
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
        if (
            buy_date.to_pydatetime().isoweekday() != DAYS["sunday"]
        ):  # buy isn't on a sunday
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
            if day_of_week == DAYS["sunday"]:  # no sells allowed on sundays
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

    def set_user_pref(self, author, pref, value):
        users = self.data.users
        row = users[users.author == author.id].tail(1)
        if row.empty:
            users = users.append({"author": author.id, pref: value}, ignore_index=True)
        else:
            users.at[row.index, pref] = value
        self.data.commit(users)

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

            message = remaining[0 : breakpoint + 1]
            yield message
            remaining = remaining[breakpoint + 1 :]
            last_line_end = message.rfind("\n")
            if last_line_end != -1 and len(message) > last_line_end + 1:
                last_line_start = last_line_end + 1
            else:
                last_line_start = 0
            if message[last_line_start] == ">":
                remaining = f"> {remaining}"

        yield remaining

    @property
    def commands(self):
        """Returns a list of commands supported by this bot."""
        return self._commands

    async def process(self, message):
        """Process a command message."""
        tokens = message.content.split(" ")
        request, params = tokens[0].lstrip("!").lower(), tokens[1:]
        params = list(filter(None, params))  # ignore any empty string parameters
        if not request:
            return
        matching = [command for command in self.commands if command.startswith(request)]
        if not matching:
            await message.channel.send(s("not_a_command", request=request), file=None)
            return
        if len(matching) > 1 and request not in matching:
            possible = ", ".join(f"!{m}" for m in matching)
            await message.channel.send(s("did_you_mean", possible=possible), file=None)
        else:
            command = request if request in matching else matching[0]
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
                            if attachment is not None
                            and i == last_page_index
                            and n == last_reply_index
                            else None
                        )
                        await message.channel.send(page, file=file)
                elif isinstance(reply, discord.embeds.Embed):
                    file = (
                        attachment
                        if attachment is not None and n == last_reply_index
                        else None
                    )
                    await message.channel.send(embed=reply, file=file)
                else:
                    raise RuntimeError("non-string non-embed reply not supported")

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

    # Any method of this class with a name that is decorated by @command is detected as a
    # bot command. These methods should have a signature like:
    #
    #     @command
    #     def command_name(self, channel, author, params)
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
    # message for the command. To document commands with parameters use a @ to delimit
    # the help message from the parameter documentation. For example:
    #
    #     """This is the help message for your command. @ [and] [these] [are] [params]"""
    #
    # A [parameter] is optional whereas a <parameter> is required.

    @command
    def help(self, channel, author, params):
        """
        Shows this help screen.
        """
        usage = "__**Turbot Help!**__\n"
        for command in self.commands:
            method = getattr(self, command)
            doc = method.__doc__.split("@")
            use, params = doc[0], ", ".join([param.strip() for param in doc[1:]])
            use = inspect.cleandoc(use)
            use = use.replace("\n", " ")

            title = f"**!{command}**"
            if params:
                title = f"{title} _{params}_"
            usage += f"\n{title}"
            usage += f"\n>  {use}"
            usage += "\n"
        usage += "\n_Turbot created by TheAstropath_"
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
        day_offset = DAYS[day_of_week] % 7
        hour_offset = 13 if time_of_day == "evening" else 0
        return start + timedelta(days=day_offset, hours=hour_offset)

    @command
    def sell(self, channel, author, params):
        """
        Log the price that you can sell turnips for on your island.
        @ <price> [day time-of-day]
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

    @command
    def buy(self, channel, author, params):
        """
        Log the price that you can buy turnips from Daisy Mae on your island.
        @ <price> [day time]
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

    @command
    def reset(self, channel, author, params):
        """
        Only Turbot Admin members can run this command. Generates a final graph for use
        with !lastweek and resets all data for all users.
        """
        if not is_turbot_admin(channel, author):
            return s("not_admin"), None

        self.generate_graph(channel, None, LASTWEEKCMD_FILE)
        prices = self.data.prices
        self.backup_prices(prices)

        buys = prices[prices.kind == "buy"].sort_values(by="timestamp")
        idx = buys.groupby(by="author")["timestamp"].idxmax()
        prices = buys.loc[idx]
        self.data.commit(prices)
        return s("reset"), None

    @command
    def lastweek(self, channel, author, params):
        """
        Displays the final graph from the last week before the data was reset.
        """
        if not Path(LASTWEEKCMD_FILE).exists():
            return s("lastweek_none"), None
        return s("lastweek"), discord.File(LASTWEEKCMD_FILE)

    @command
    def graph(self, channel, author, params):
        """
        Generates a historical graph of turnip prices for all users.
        """
        success = self.generate_graph(channel, None, GRAPHCMD_FILE)
        attach = discord.File(GRAPHCMD_FILE) if success else None
        return s("graph_all_users"), attach

    @command
    def history(self, channel, author, params):
        """
        Show the historical turnip prices for a user. If no user is specified, it will
        display your own prices. @ [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        prices = self.data.prices
        yours = prices[prices.author == target_id]
        lines = [s("history_header", name=target_name)]
        for _, row in yours.iterrows():
            time = self.to_usertime(target_id, row.timestamp)
            lines.append(
                s(
                    f"history_{row.kind}",
                    price=row.price,
                    timestamp=h(time),
                    day_and_time=day_and_time(time),
                )
            )
        return "\n".join(lines), None

    @command
    def oops(self, channel, author, params):
        """
        Remove your last logged turnip price.
        """
        target = author.id
        target_name = discord_user_name(channel, target)
        prices = self.data.prices
        prices = prices.drop(prices[prices.author == author.id].tail(1).index)
        self.data.commit(prices)
        return s("oops", name=target_name), None

    @command
    def clear(self, channel, author, params):
        """
        Clears all of your own historical turnip prices.
        """
        user_id = discord_user_id(channel, str(author))
        prices = self.data.prices
        prices = prices[prices.author != user_id]
        self.data.commit(prices)
        return s("clear", name=author), None

    @command
    def best(self, channel, author, params):
        """
        Finds the best, most recent, buy or sell price currently available.
        The default is to look for the best sell.
        @ [buy|sell]
        """
        kind = params[0].lower() if params else "sell"
        if kind not in ["sell", "buy"]:
            return s("best_invalid_param"), None

        prices = self.data.prices
        past = datetime.now(pytz.utc) - timedelta(hours=12)
        sells = prices[(prices.kind == kind) & (prices.timestamp > past)]
        idx = sells.groupby(by="author").price.transform(max) == sells.price
        bests = sells[idx].sort_values(by="price", ascending=kind == "buy")
        lines = [s(f"best_{kind}_header")]
        for _, row in bests.iterrows():
            name = discord_user_from_id(channel, row.author)
            timestamp = h(self.to_usertime(row.author, row.timestamp))
            lines.append(s("best", name=name, price=row.price, timestamp=timestamp))
        return "\n".join(lines), None

    @command
    def collect(self, channel, author, params):
        """
        Mark collectables as donated to your museum. The names must match the in-game item
        name exactly. @ <comma, separated, list, of, things>
        """
        if not params:
            return s("collect_no_params"), None

        items = set(sanitize_collectable(item) for item in " ".join(params).split(","))
        valid, invalid = self.assets.validate(items)

        lines = []

        def add_lines(kind, valid_items, fullset):
            if not valid_items:
                return
            store = getattr(self.data, kind)
            yours = store[store.author == author.id]
            dupes = yours.loc[yours.name.isin(valid_items)].name.values.tolist()
            new_items = list(set(valid_items) - set(dupes))
            new_data = [[author.id, name] for name in new_items]
            new_fossils = pd.DataFrame(columns=store.columns, data=new_data)
            store = store.append(new_fossils, ignore_index=True)
            yours = store[store.author == author.id]  # re-fetch for congrats
            self.data.commit(store)
            if new_items:
                lines.append(s(f"collect_{kind}_new", items=", ".join(sorted(new_items))))
            if dupes:
                lines.append(s(f"collect_{kind}_dupe", items=", ".join(sorted(dupes))))
            if len(fullset) == len(yours.index):
                lines.append(s(f"congrats_all_{kind}"))

        for kind in self.assets.collectables:
            add_lines(kind, valid[kind], self.assets[kind].all)

        if invalid:
            lines.append(s("invalid_collectable", items=", ".join(sorted(invalid))))

        return "\n".join(lines), None

    @command
    def uncollect(self, channel, author, params):
        """
        Unmark collectables as donated to your museum. The names must match the in-game
        item name exactly. @ <comma, separated, list, of, things>
        """
        if not params:
            return s("uncollect_no_params"), None

        items = set(sanitize_collectable(item) for item in " ".join(params).split(","))
        valid, invalid = self.assets.validate(items)

        lines = []

        def add_lines(kind, valid_items):
            store = getattr(self.data, kind)
            yours = store[store.author == author.id]
            previously_collected = yours.loc[yours.name.isin(valid_items)]
            deleted = set(previously_collected.name.values.tolist())
            didnt_have = valid_items - deleted
            store = store.drop(previously_collected.index)
            self.data.commit(store)
            if deleted:
                items_str = ", ".join(sorted(deleted))
                lines.append(s(f"uncollect_{kind}_deleted", items=items_str))
            if didnt_have:
                items_str = ", ".join(sorted(didnt_have))
                lines.append(s(f"uncollect_{kind}_already", items=items_str))

        for kind in self.assets.collectables:
            add_lines(kind, valid[kind])

        if invalid:
            lines.append(s("invalid_collectable", items=", ".join(sorted(invalid))))

        return "\n".join(lines), None

    @command
    def search(self, channel, author, params):
        """
        Searches all users to see who needs the given collectables. The names must match
        the in-game item name, and more than one can be provided if separated by commas.
        @ <comma, separated, list, of, collectables>
        """
        if not params:
            return s("search_no_params"), None

        items = set(sanitize_collectable(item) for item in " ".join(params).split(","))
        valid, invalid = self.assets.validate(items)

        def get_results(kind, valid_items):
            store = getattr(self.data, kind)
            users = store.author.unique()
            results = defaultdict(list)
            for collected_item in valid_items:
                havers = store[store.name == collected_item].author.unique()
                needers = np.setdiff1d(users, havers).tolist()
                for needer in needers:
                    name = discord_user_from_id(channel, needer)
                    results[name].append(collected_item)
            return results

        results = {}
        for kind in self.assets.collectables:
            results[kind] = get_results(kind, valid[kind])

        if not any(result for result in results.values()) and not invalid:
            return s("search_all_not_needed"), None

        searched = set()
        for valid_items in valid.values():
            searched.update(valid_items)

        needed = set()
        for result in results.values():
            for items in result.values():
                needed.update(items)

        not_needed = searched - needed

        lines = []
        for kind, result in results.items():
            for name, items in result.items():
                items_str = ", ".join(sorted(items))
                lines.append(s(f"search_{kind}_row", name=name, items=items_str))
        if not_needed:
            items_str = ", ".join(sorted(not_needed))
            lines.append(s("search_not_needed", items=items_str))
        if invalid:
            lines.append(s("search_invalid", items=", ".join(sorted(invalid))))
        return "\n".join(sorted(lines)), None

    @command
    def uncollected(self, channel, author, params):
        """
        Lists all collectables that you still need to donate. If a user is provided, it
        gives the same information for that user instead. @ [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        lines = []

        def add_lines(kind, fullset):
            store = getattr(self.data, kind)
            your_items = store[store.author == target_id]
            collected_items = set(your_items.name.unique())
            remaining_items = fullset - collected_items
            if remaining_items:
                count_key = f"uncollected_{kind}_count"
                remaining_key = f"uncollected_{kind}_remaining"
                count = len(remaining_items)
                items_str = ", ".join(sorted(remaining_items))
                lines.append(s(count_key, count=count, name=target_name))
                lines.append(s(remaining_key, items=items_str))
            else:
                lines.append(s(f"congrats_all_{kind}"))

        for kind in self.assets.collectables:
            add_lines(kind, self.assets[kind].all)

        return "\n".join(lines), None

    @command
    def needed(self, channel, author, params):
        """
        Lists all the needed items for all the channel members. As the only parameter
        give the name of the kind of collectable to return.
        @ <fossils|bugs|fish|art|songs>
        """
        if not params:
            return s("needed_no_param"), None

        kind = params[0].lower()
        if kind not in ["fossils", "bugs", "fish", "art", "songs"]:
            return s("needed_invalid_param"), None

        fullset = self.assets[kind].all
        store = getattr(self.data, kind)
        authors = [member.id for member in channel.members if member.id != self.user.id]
        total = pd.DataFrame(list(product(authors, fullset)), columns=store.columns)
        merged = total.merge(store, indicator=True, how="outer")
        needed = merged[merged["_merge"] == "left_only"]

        limit = 10
        lines = []

        for user, df in needed.groupby(by="author"):
            name = discord_user_name(channel, user)
            items_list = sorted([row["name"] for _, row in df.iterrows()])
            if len(items_list) == len(fullset):
                continue
            elif len(items_list) > limit:
                items_str = s("needed_lots", limit=limit, kind=kind)
            else:
                items_str = ", ".join(items_list)
            lines.append(s("needed", name=name, items=items_str))

        if not lines:
            return s("needed_none", kind=kind), None

        return "\n".join(sorted(lines)), None

    @command
    def collected(self, channel, author, params):
        """
        Lists all collectables that you have already donated. If a user is provided, it
        gives the same information for that user instead. @ [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        def get_collection(kind):
            store = getattr(self.data, kind)
            your_items = store[store.author == target_id]
            return set(your_items.name.unique())

        collected_items = {
            kind: get_collection(kind) for kind in self.assets.collectables
        }

        all_items = {
            kind: len(collected_items[kind]) == len(self.assets[kind].all)
            for kind in self.assets.collectables
        }

        lines = []

        if any(flag for flag in all_items.values()):
            for kind, flag in all_items.items():
                if flag:
                    lines.append(s(f"congrats_all_{kind}"))
            if all(flag for flag in all_items.values()):
                return "\n".join(lines), None

        for kind, items in collected_items.items():
            if items and not all_items[kind]:
                key = f"collected_{kind}"
                count = len(items)
                items_str = ", ".join(sorted(items))
                lines.append(s(key, name=target_name, count=count, items=items_str))

        return "\n".join(lines), None

    @command
    def predict(self, channel, author, params):
        """
        Get a link to a prediction calculator for a price history. @ [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        target_user = discord_user_from_id(channel, target_id)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        timeline = self.get_user_timeline(target_id)
        if not timeline[0]:
            return s("cant_find_buy", name=target_name), None

        success = self.generate_graph(channel, target_user, GRAPHCMD_FILE)
        query = ".".join((str(price) if price else "") for price in timeline).rstrip(".")
        url = f"{self.base_prophet_url}{query}"
        attach = discord.File(GRAPHCMD_FILE) if success else None
        return s("predict", name=target_name, url=url), attach

    @command
    def pref(self, channel, author, params):
        """
        Set one of your user preferences. @ <preference> <value>
        """
        if not params:
            return s("pref_no_params", prefs=", ".join(USER_PREFRENCES)), None

        pref = params[0].lower()
        if pref not in USER_PREFRENCES:
            return s("pref_invalid_pref", prefs=", ".join(USER_PREFRENCES)), None
        if len(params) <= 1:
            return s("pref_no_value", pref=pref), None

        value = " ".join(params[1:])
        validated_value = getattr(Validate, pref)(value)
        if not validated_value:
            return s(f"{pref}_invalid"), None

        self.set_user_pref(author, pref, validated_value)
        return s("pref", pref=pref, name=author), None

    @command
    def count(self, channel, author, params):
        """
        Provides a count of the number of pieces of collectables for the comma-separated
        list of users. @ [comma, separated, list, of, users]
        """
        if not params:
            user = discord_user_from_id(channel, author.id)
            valid = [(user.name, user.id)]
            invalid = []
        else:
            users = set(sanitize_collectable(p) for p in " ".join(params).split(","))
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

        def add_valid_lines(kind, fullset):
            lines.append(s(f"count_{kind}_valid_header"))
            store = getattr(self.data, kind)
            for user_name, user_id in sorted(valid):
                yours = store[store.author == user_id]
                collected = set(yours.name.unique())
                remaining = fullset - collected
                count = len(remaining)
                lines.append(s(f"count_{kind}_valid", name=user_name, count=count))

        if valid:
            for kind in self.assets.collectables:
                add_valid_lines(kind, self.assets[kind].all)

        if invalid:
            lines.append(s("count_invalid_header"))
            for user in invalid:
                lines.append(s("count_invalid", name=user))

        return "\n".join(lines), None

    @command
    def art(self, channel, author, params):
        """
        Get info about pieces of art that are available
        @ [comma, separated, list, of, art, pieces]
        """
        response = ""
        if params:
            items = set(
                sanitize_collectable(item) for item in " ".join(params).split(",")
            )
            validset, invalidset = self.assets.validate(items, kinds=["art"])
            art_data = self.assets["art"].data
            valid = sorted(list(validset["art"]))
            invalid = sorted(list(invalidset))
            lines = []
            response = s("art_header") + "\n"
            for art in valid:
                piece = art_data[art_data.name == art].iloc[0]
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
            items = self.assets["art"].all
            response = s("allart", list=", ".join(sorted(items)))

        return response, None

    def creatures_available_now(self, now, df):
        """Returns the names of creatures in df that are available right now."""
        for _, row in df.iterrows():
            time = row["time"]
            if time.lower() == "all day":
                yield row["name"]
            else:
                ranges = time.lower().split("&")
                for r in ranges:
                    lhs, rhs = [t.strip().lower() for t in r.split("-")]
                    lhs_hour, lhs_ampm = lhs.split(" ")
                    rhs_hour, rhs_ampm = rhs.split(" ")
                    lhs_hour, rhs_hour = int(lhs_hour), int(rhs_hour)
                    now_hour = now.hour
                    now_ampm = "am" if now_hour < 12 else "pm"

                    # | lhs_ampm | rhs_ampm | now_ampm | yield if (use _hour vars) |
                    # | -------- | -------- | -------- | ------------------------- |
                    # | am       | am       | am       | lhs <= now <= rhs         |
                    # | am       | am       | pm       | False                     |
                    # | am       | pm       | am       | lhs <= now <= rhs + 12    |
                    # | am       | pm       | pm       | lhs <= now <= rhs + 12    |
                    # | pm       | am       | am       | now <= rhs                |
                    # | pm       | am       | pm       | now >= lhs                |
                    # | pm       | pm       | am       | False                     |
                    # | pm       | pm       | pm       | lhs <= now <= rhs         |
                    if lhs_ampm == "am" and rhs_ampm == "am" and now_ampm == "am":
                        if lhs_hour <= now_hour <= rhs_hour:
                            yield row["name"]
                    elif lhs_ampm == "am" and rhs_ampm == "am" and now_ampm == "pm":
                        continue
                    elif lhs_ampm == "am" and rhs_ampm == "pm" and now_ampm == "am":
                        if lhs_hour <= now_hour <= rhs_hour + 12:
                            yield row["name"]
                    elif lhs_ampm == "am" and rhs_ampm == "pm" and now_ampm == "pm":
                        if lhs_hour <= now_hour <= rhs_hour + 12:
                            yield row["name"]
                    elif lhs_ampm == "pm" and rhs_ampm == "am" and now_ampm == "am":
                        if now_hour <= rhs_hour:
                            yield row["name"]
                    elif lhs_ampm == "pm" and rhs_ampm == "am" and now_ampm == "pm":
                        if now_hour >= lhs_hour:
                            yield row["name"]
                    elif lhs_ampm == "pm" and rhs_ampm == "pm" and now_ampm == "am":
                        continue
                    else:  # lhs_ampm == "pm" and rhs_ampm == "pm" and now_ampm == "pm"
                        if lhs_hour <= now_hour <= rhs_hour:
                            yield row["name"]

    def _creatures(self, *, author, params, kind, source, force_text=False):
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
        else:
            if kind == "fish":
                caught = self.data.fish
                caught = caught[caught.author == author.id]
            else:  # kind == "bugs"
                caught = self.data.bugs
                caught = caught[caught.author == author.id]

            if caught is not None and not caught.empty:
                already = available.loc[available.name.isin(caught.name)]
                available = available.drop(already.index)

            found = available

            if found.empty:
                return s(f"{kind}_none_available")

        now_names = set(self.creatures_available_now(now, found))
        found_now = found[found.name.isin(now_names)]
        found_this_month = found[~found.name.isin(now_names)]

        def get_response(df, force_text):
            if force_text or len(df) > EMBED_LIMIT:
                lines = [s(kind, **details(row)) for _, row in df.iterrows()]
                return "\n".join(sorted(lines))

            response = []
            rows = [row for _, row in df.iterrows()]
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
            return response

        response_now = get_response(found_now, force_text=force_text)
        response_this_month = get_response(found_this_month, force_text=True)

        responses = []

        if response_now:
            header = s("creature_available_now", kind=kind.title())
            if isinstance(response_now, str):
                responses.append(f"{header}\n{response_now}")
            else:
                responses.append(header)
                responses.extend(response_now)

        if response_this_month:
            # response_this_month is always forced to be text
            header = s("creature_available_this_month", kind=kind.title())
            responses.append(f"{header}\n{response_this_month}")

        def add_header(responses):
            if kind != "bugs":
                return responses
            if random.randint(0, 100) > 70:
                responses.insert(0, s("bugs_header"))
            return responses

        return add_header(responses)

    @command
    def fish(self, channel, author, params):
        """
        Tells you what fish are available now in your hemisphere.
        @ [name|leaving|arriving]
        """
        source = self.assets["fish"].data
        found = self._creatures(author=author, params=params, kind="fish", source=source)
        return found, None

    @command
    def bugs(self, channel, author, params):
        """
        Tells you what bugs are available now in your hemisphere.
        @ [name|leaving|arriving]
        """
        source = self.assets["bugs"].data
        found = self._creatures(author=author, params=params, kind="bugs", source=source)
        return found, None

    @command
    def new(self, channel, author, params):
        """
        Tells you what new things available in your hemisphere right now.
        """
        bugs_data = self.assets["bugs"].data
        bugs = self._creatures(
            author=author,
            params=["arriving"],
            kind="bugs",
            source=bugs_data,
            force_text=True,
        )
        fish_data = self.assets["fish"].data
        fish = self._creatures(
            author=author,
            params=["arriving"],
            kind="fish",
            source=fish_data,
            force_text=True,
        )
        return [*bugs, *fish], None

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

    @command
    def info(self, channel, author, params):
        """
        Gives you information on a user. @ [user]
        """
        if len(params) < 1:
            return s("info_no_params"), None

        query = " ".join(params).lower()  # allow spaces in names

        for _, row in self.data.users.iterrows():
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

    @command
    def about(self, channel, author, params):
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
    """Returns the authorized channels the environment or your channels config file."""
    channels = getenv("TURBOT_CHANNELS", None)
    if channels:
        return channels.split(";")

    try:
        with open(channels_file, "r") as channels_file:
            return [line.strip() for line in channels_file.readlines()]
    except IOError:
        return []


def apply_migrations():  # pragma: no cover
    """Applies any migration scripts that haven't already been applied."""

    def migration_key(script):
        return int(script.name[0:3])

    applied = set()
    if MIGRATIONS_FILE.exists():
        with open(MIGRATIONS_FILE) as f:
            applied.update(line.strip() for line in f.readlines())
    with open(MIGRATIONS_FILE, "a") as f:
        for migration in sorted(MIGRATIONS_DIR.glob("*.py"), key=migration_key):
            if migration.name not in applied:
                print(f"applying migration {migration.name}...", end=" ")
                cmd = ["python3", migration]
                proc = run(cmd, capture_output=True)
                if proc.returncode != 0:
                    print()
                    print(proc.stdout.decode("utf-8"))
                    print(proc.stderr.decode("utf-8"), file=sys.stderr)
                    sys.exit(1)
                print("done")
                f.write(f"{migration.name}\n")


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
    help=(
        "authorize a channel; use this multiple times to authorize multiple channels; "
        "you can also set the list of channels via the TURBOT_CHANNELS environment "
        "variable separated by ; (semicolon)."
    ),
)
@click.option(
    "-a",
    "--auth-channels-file",
    default=DEFAULT_CONFIG_CHANNELS,
    help=(
        "read channel names from this file; you can also set the list of channels via "
        "the TURBOT_CHANNELS environment variable separated by ; (semicolon)."
    ),
)
@click.option(
    "--db-dir", default=DEFAULT_DB_DIR, help="use this directory for user db files"
)
@click.version_option(version=__version__)
@click.option(
    "--dev",
    default=False,
    is_flag=True,
    help="Development mode, automatically reload bot when source changes",
)
def main(
    log_level, verbose, bot_token_file, channel, auth_channels_file, db_dir, dev,
):  # pragma: no cover
    auth_channels = get_channels(auth_channels_file) + list(channel)
    if not auth_channels:
        print("error: you must provide at least one authorized channel", file=sys.stderr)
        sys.exit(1)

    # ensure transient application directories exist
    db_dir.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    client = Turbot(
        token=get_token(bot_token_file),
        channels=auth_channels,
        db_dir=db_dir,
        log_level=getattr(logging, "DEBUG" if verbose else log_level),
    )

    apply_migrations()

    if dev:
        reloader = hupper.start_reloader("turbot.main")
        reloader.watch_files(client.assets.files())

    client.run()


if __name__ == "__main__":
    main()
