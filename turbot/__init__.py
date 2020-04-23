#!/usr/bin/env python

import inspect
import logging
import random
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from os.path import dirname, realpath
from pathlib import Path
from string import Template

import click
import discord
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

matplotlib.use("Agg")

PACKAGE_ROOT = Path(dirname(realpath(__file__)))
RUNTIME_ROOT = Path(".")

# application configuration files
CONFIG_DIR = RUNTIME_ROOT / "config"
DEFAULT_CONFIG_TOKEN = CONFIG_DIR / "token.txt"
DEFAULT_CONFIG_CHANNELS = CONFIG_DIR / "channels.txt"

# static application asset data
DATA_DIR = PACKAGE_ROOT / "data"
STRINGS_DATA_FILE = DATA_DIR / "strings.yaml"
FOSSILS_DATA_FILE = DATA_DIR / "fossils.txt"
FISH_DATA_FILE = DATA_DIR / "fish.csv"
BUGS_DATA_FILE = DATA_DIR / "bugs.csv"

# persisted user and application data
DB_DIR = RUNTIME_ROOT / "db"
DEFAULT_DB_FOSSILS = DB_DIR / "fossils.csv"
DEFAULT_DB_PRICES = DB_DIR / "prices.csv"
DEFAULT_DB_USERS = DB_DIR / "users.csv"


# temporary application files
TMP_DIR = RUNTIME_ROOT / "tmp"
GRAPHCMD_FILE = TMP_DIR / "graphcmd.png"
LASTWEEKCMD_FILE = TMP_DIR / "lastweek.png"

# ensure application directories exist
CONFIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

with open(STRINGS_DATA_FILE) as f:
    STRINGS = load(f, Loader=Loader)

with open(FOSSILS_DATA_FILE) as f:
    FOSSILS = frozenset([line.strip().lower() for line in f.readlines()])

FISH = pd.read_csv(FISH_DATA_FILE)
BUGS = pd.read_csv(BUGS_DATA_FILE)


def s(key, **kwargs):
    """Returns a string from data/strings.yaml with subsitutions."""
    data = STRINGS.get(key, "")
    return Template(data).substitute(kwargs)


def discord_user_from_name(channel, name):
    """Returns the discord user from the given channel and name."""
    if not name:
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
        self, token, channels, prices_file, fossils_file, users_file, log_level=None
    ):
        if log_level:
            logging.basicConfig(level=log_level)
        super().__init__()
        self.token = token
        self.channels = channels
        self.prices_file = prices_file
        self.fossils_file = fossils_file
        self.users_file = users_file
        self.base_prophet_url = "https://turnipprophet.io/?prices="  # TODO: configurable?
        self._prices_data = None  # do not use directly, load it from load_prices()
        self._fossils_data = None  # do not use directly, load it from load_fossils()
        self._users_data = None  # do not use directly, load it from load_users()
        self._last_backup_filename = None

    def run(self):
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
        if self._users_data is None:
            try:
                self._users_data = pd.read_csv(self.users_file)
            except FileNotFoundError:
                self._users_data = pd.DataFrame(
                    columns=["author", "hemisphere", "timezone"]
                )
        return self._users_data

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

    def generate_graph(self, channel, user, graphname):
        """Generates a nice looking graph of user data."""
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

        if user:  # graph specific user
            dates = []
            prices = []
            userId = discord_user_id(channel, user)
            userName = discord_user_name(channel, userId)
            if not userId or not userName:
                return False
            legendElems.append(userName)
            yours = priceList.loc[priceList.author == userId]
            for _, row in yours.iterrows():
                if row.kind == "sell":
                    prices.append(row.price)
                    dates.append(row.timestamp)
            if dates:
                plt.plot(dates, prices, linestyle="-", marker="o", label=userName)
        else:  # graph all users
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
                return False

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
        plt.close("all")

        return True

    def append_price(self, author, kind, price):
        """Adds a price to the prices data file for the given author and kind."""
        prices = self.load_prices()
        prices = prices.append(
            pd.DataFrame(
                columns=prices.columns,
                data=[[author.id, kind, price, datetime.now(pytz.utc)]],
            ),
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

    def _get_user_prefs(self, author_id):
        users = self.load_users()
        row = users[users.author == author_id].tail(1)
        if row.empty:
            return {}
        return row.to_dict(orient="records")[0]

    def get_user_hemisphere(self, author_id):
        prefs = self._get_user_prefs(author_id)
        if (
            not prefs
            or "hemisphere" not in prefs
            or not prefs["hemisphere"]
            or not isinstance(prefs["hemisphere"], str)
        ):
            return None
        return prefs["hemisphere"]

    def get_user_timezone(self, author_id):
        prefs = self._get_user_prefs(author_id)
        if (
            not prefs
            or "timezone" not in prefs
            or not prefs["timezone"]
            or not isinstance(prefs["timezone"], str)
        ):
            return pytz.UTC
        return pytz.timezone(prefs["timezone"])

    def to_usertime(self, author_id, dt):
        return dt.tz_convert(self.get_user_timezone(author_id))

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

            yield remaining[0:breakpoint]
            remaining = remaining[breakpoint + 1 :]

        yield remaining

    async def process(self, message):
        """Process a command message."""
        tokens = message.content.split(" ")
        request, params = tokens[0].lstrip("!"), tokens[1:]
        if not request:
            return
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        commands = [member[0] for member in members if member[0].endswith("_command")]
        matching = [command for command in commands if command.startswith(request)]
        if not matching:
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
            pages = list(self.paginate(response))
            last_page_index = len(pages) - 1
            for i, page in enumerate(pages):
                file = attachment if attachment and i == last_page_index else None
                await message.channel.send(page, file=file)

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

    # Any method of this class that end in the name _command are automatically
    # detected as bot commands. The methods should have a signature like:
    #
    #     def your_command(self, channel, author, params)
    #
    # - `channel` is the Discord channel where the command message was sent.
    # - `author` is the Discord author who sent the command.
    # - `params` are any space delimitered parameters also sent with the command.
    #
    # The return value for a command method should be `(string, discord.File)` where the
    # string is the response message the bot should send to the channel and the file
    # object is an attachment to send with the message. For no attachment, use `None`.
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

    def sell_command(self, channel, author, params):
        """
        Log the price that you can sell turnips for on your island. | <price>
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

        logging.debug("saving sell price of %s bells for user id %s", price, author.id)
        self.append_price(author, "sell", price)

        key = (
            "sell_new_price"
            if not last_price
            else "sell_higher_price"
            if price > last_price
            else "sell_lower_price"
            if price < last_price
            else "sell_same_price"
        )
        return s(key, price=price, name=author, last_price=last_price), None

    def buy_command(self, channel, author, params):
        """
        Log the price that you can buy turnips from Daisy Mae on your island. | <price>
        """
        if not params:
            return s("buy_no_params"), None

        price = params[0]
        if not price.isnumeric():
            return s("buy_nonnumeric_price"), None

        price = int(price)
        if price <= 0:
            return s("buy_nonpositive_price"), None

        logging.debug("saving buy price of %s bells for user id %s", price, author.id)
        self.append_price(author, "buy", price)
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
        if not user_id or not user_name:
            return s("cant_find_user", name=params[0]), None

        self.generate_graph(channel, user_name, GRAPHCMD_FILE)
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
        if 4 in patterns:
            lines.append(s("turnippattern_pattern4"))
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
                    timestamp=self.to_usertime(target_id, row.timestamp),
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
                    timestamp=self.to_usertime(row.author, row.timestamp),
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
        Mark fossils as donated to your museum. The names must match the in-game item
        name, and more than one can be provided if separated by commas.
        | <list of fossils>
        """
        if not params:
            return s("collect_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))
        valid = items.intersection(FOSSILS)
        invalid = items.difference(FOSSILS)

        fossils = self.load_fossils()
        yours = fossils[fossils.author == author.id]
        dupes = yours.loc[yours.name.isin(valid)].name.values.tolist()
        new_names = list(set(valid) - set(dupes))
        new_data = [[author.id, name] for name in new_names]
        new_fossils = pd.DataFrame(columns=fossils.columns, data=new_data)
        fossils = fossils.append(new_fossils, ignore_index=True)
        self.save_fossils(fossils)

        lines = []
        if new_names:
            lines.append(s("collect_new", items=", ".join(sorted(new_names))))
        if dupes:
            lines.append(s("collect_dupe", items=", ".join(sorted(dupes))))
        if invalid:
            lines.append(s("collect_bad", items=", ".join(sorted(invalid))))
        return "\n".join(lines), None

    def uncollect_command(self, channel, author, params):
        """
        Unmark fossils as donated to your museum. The names must match the in-game item
        name, and more than one can be provided if separated by commas.
        | <list of fossils>
        """
        if not params:
            return s("uncollect_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))
        valid = items.intersection(FOSSILS)
        invalid = items.difference(FOSSILS)

        fossils = self.load_fossils()
        yours = fossils[fossils.author == author.id]
        previously_collected = yours.loc[yours.name.isin(valid)]
        deleted = set(previously_collected.name.values.tolist())
        didnt_have = valid - deleted
        fossils = fossils.drop(previously_collected.index)
        self.save_fossils(fossils)

        lines = []
        if deleted:
            lines.append(s("uncollect_deleted", items=", ".join(sorted(deleted))))
        if didnt_have:
            lines.append(s("uncollect_already", items=", ".join(sorted(didnt_have))))
        if invalid:
            lines.append(s("fossil_bad", items=", ".join(sorted(invalid))))
        return "\n".join(lines), None

    def fossilsearch_command(self, channel, author, params):
        """
        Searches all users to see who needs the listed fossils. The names must match the
        in-game item name, and more than one can be provided if separated by commas.
        | <list of fossils>
        """
        if not params:
            return s("fossilsearch_no_params"), None

        items = set(item.strip().lower() for item in " ".join(params).split(","))
        valid = items.intersection(FOSSILS)
        invalid = items - valid

        fossils = self.load_fossils()
        users = fossils.author.unique()
        results = defaultdict(list)
        for fossil in valid:
            havers = fossils[fossils.name == fossil].author.unique()
            needers = np.setdiff1d(users, havers).tolist()
            for needer in needers:
                name = discord_user_from_id(channel, needer)
                results[name].append(fossil)

        if not results and not invalid:
            return s("fossilsearch_noneed"), None

        if not results and invalid:
            lines = [s("fossilsearch_header")]
            if valid:
                lines.append(
                    s("fossilsearch_row", name="No one", fossils=", ".join(sorted(valid)))
                )
            lines.append(s("fossil_bad", items=", ".join(sorted(invalid))))
            return "\n".join(lines), None

        lines = [s("fossilsearch_header")]
        for name, needed in results.items():
            need_list = fossils = ", ".join(sorted(needed))
            lines.append(s("fossilsearch_row", name=name, fossils=need_list))
        if invalid:
            lines.append(s("fossil_bad", items=", ".join(sorted(invalid))))
        return "\n".join(lines), None

    def allfossils_command(self, channel, author, params):
        """
        Shows all possible fossils that you can donate to the museum.
        """
        return s("allfossils", list=", ".join(sorted(FOSSILS))), None

    def listfossils_command(self, channel, author, params):
        """
        Lists all fossils that you still need to donate. If a user is provided, it gives
        the same information for that user instead. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        fossils = self.load_fossils()
        yours = fossils[fossils.author == target_id]
        collected = set(yours.name.unique())
        remaining = FOSSILS - collected
        return (
            s(
                "listfossils",
                name=target_name,
                count=len(remaining),
                items=", ".join(sorted(remaining)),
            ),
            None,
        )

    def collectedfossils_command(self, channel, author, params):
        """
        Lists all fossils that you have already donated. If a user is provided, it
        gives the same information for that user instead. | [user]
        """
        target = author.id if not params else params[0]
        target_name = discord_user_name(channel, target)
        target_id = discord_user_id(channel, target_name)
        if not target_name or not target_id:
            return s("cant_find_user", name=target), None

        fossils = self.load_fossils()
        yours = fossils[fossils.author == target_id]
        collected = set(yours.name.unique())

        return (
            s(
                "collectedfossils",
                name=target_name,
                count=len(collected),
                items=", ".join(sorted(collected)),
            ),
            None,
        )

    def fossilcount_command(self, channel, author, params):
        """
        Provides a count of the number of fossils remaining for the comma-separated list
        of users. | <list of users>
        """
        if not params:
            return s("fossilcount_no_params"), None

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
            lines.append(s("fossilcount_valid_header"))
            fossils = self.load_fossils()
            for user_name, user_id in sorted(valid):
                yours = fossils[fossils.author == user_id]
                collected = set(yours.name.unique())
                remaining = FOSSILS - collected
                lines.append(s("fossilcount_valid", name=user_name, count=len(remaining)))
        if invalid:
            lines.append(s("fossilcount_invalid_header"))
            for user in invalid:
                lines.append(s("fossilcount_invalid", name=user))
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

        prices = self.load_prices()
        past = datetime.now(pytz.utc) - timedelta(days=12)
        yours = prices[(prices.author == target_id) & (prices.timestamp > past)]
        yours = yours.sort_values(by=["timestamp"])

        # convert all timestamps to the target user's timezone
        target_timezone = self.get_user_timezone(target_id)
        yours["timestamp"] = yours.timestamp.dt.tz_convert(target_timezone)

        recent_buy = yours[yours.kind == "buy"].tail(1)
        if recent_buy.empty:
            return s("cant_find_buy", name=target_name), None

        buy_date = recent_buy.timestamp.iloc[0]
        buy_price = recent_buy.price.iloc[0]

        sells = yours[yours.kind == "sell"]
        groups = sells.set_index("timestamp").groupby(pd.Grouper(freq="D"))
        sell_data = {}
        for day, df in groups:
            days_since_buy = (day - buy_date).days
            sell_data[days_since_buy] = df.price.iloc[0:2]

        sequence = [""] * 12
        for day in range(0, 6):
            if day in sell_data and sell_data[day].any():
                sequence[day * 2] = sell_data[day][0]
                if len(sell_data[day]) > 1:
                    sequence[day * 2 + 1] = sell_data[day][1]

        query = f"{buy_price}.{'.'.join(str(i) for i in sequence)}".rstrip(".")
        url = f"{self.base_prophet_url}{query}"
        return s("predict", name=target_name, url=url), None

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

    def fish_command(self, channel, author, params):
        """
        Tell you what fish are available now in your hemisphere. | [name|leaving]
        """
        hemisphere = self.get_user_hemisphere(author.id)
        if not hemisphere:
            return s("no_hemisphere"), None

        now = datetime.now(pytz.utc)
        this_month = now.strftime("%b").lower()
        next_month = (now + timedelta(days=33)).strftime("%b").lower()
        last_month = (now - timedelta(days=33)).strftime("%b").lower()
        available = FISH[(FISH.hemisphere == hemisphere) & (FISH[this_month] == 1)]

        def details(row):
            alert = (
                "**GONE NEXT MONTH!**"
                if not row[next_month]
                else "_New this month_"
                if not row[last_month]
                else ""
            )
            months = ", ".join(
                filter(
                    None,
                    [
                        "Jan" if row["jan"] else None,
                        "Feb" if row["feb"] else None,
                        "Mar" if row["mar"] else None,
                        "Apr" if row["apr"] else None,
                        "May" if row["may"] else None,
                        "Jun" if row["jun"] else None,
                        "Jul" if row["jul"] else None,
                        "Aug" if row["aug"] else None,
                        "Sep" if row["sep"] else None,
                        "Oct" if row["oct"] else None,
                        "Nov" if row["nov"] else None,
                        "Dec" if row["dec"] else None,
                    ],
                )
            )
            return {
                **row,
                "name": row["name"].capitalize(),
                "months": months,
                "alert": alert,
            }

        if params:
            search = params[0]
            if search == "leaving":
                found = available[available[next_month] == 0]
            else:
                found = available[available.name.str.contains(search)]
            if found.empty:
                return s("fish_none_found", search=search), None
            else:
                lines = [s("fish_detail", **details(row)) for _, row in found.iterrows()]
                return "\n".join(sorted(lines)), None

        lines = [s("fish", **details(row)) for _, row in available.iterrows()]
        return "\n".join(sorted(lines)), None

    def bugs_command(self, channel, author, params):
        """
        Tell you what bugs are available now in your hemisphere. | [name|leaving]
        """
        hemisphere = self.get_user_hemisphere(author.id)
        if not hemisphere:
            return s("no_hemisphere"), None

        now = datetime.now(pytz.utc)
        this_month = now.strftime("%b").lower()
        next_month = (now + timedelta(days=33)).strftime("%b").lower()
        last_month = (now - timedelta(days=33)).strftime("%b").lower()
        available = BUGS[(BUGS.hemisphere == hemisphere) & (BUGS[this_month] == 1)]

        def details(row):
            alert = (
                "**GONE NEXT MONTH!**"
                if not row[next_month]
                else "_New this month_"
                if not row[last_month]
                else ""
            )
            months = ", ".join(
                filter(
                    None,
                    [
                        "Jan" if row["jan"] else None,
                        "Feb" if row["feb"] else None,
                        "Mar" if row["mar"] else None,
                        "Apr" if row["apr"] else None,
                        "May" if row["may"] else None,
                        "Jun" if row["jun"] else None,
                        "Jul" if row["jul"] else None,
                        "Aug" if row["aug"] else None,
                        "Sep" if row["sep"] else None,
                        "Oct" if row["oct"] else None,
                        "Nov" if row["nov"] else None,
                        "Dec" if row["dec"] else None,
                    ],
                )
            )
            return {
                **row,
                "name": row["name"].capitalize(),
                "months": months,
                "alert": alert,
            }

        def add_header(lines):
            if random.randint(0, 100) > 70:
                lines.insert(0, s("bugs_header"))
            return lines

        if params:
            search = params[0]
            if search == "leaving":
                found = available[available[next_month] == 0]
            else:
                found = available[available.name.str.contains(search)]
            if found.empty:
                return s("bugs_none_found", search=search), None
            else:
                lines = sorted(
                    [s("bugs_detail", **details(row)) for _, row in found.iterrows()]
                )
                return "\n".join(add_header(lines)), None

        lines = sorted([s("bugs", **details(row)) for _, row in available.iterrows()])
        return "\n".join(add_header(lines)), None


def get_token(token_file):
    """Returns the discord token from your token config file."""
    try:
        with open(token_file, "r") as f:
            return f.readline().strip()
    except IOError as e:
        with redirect_stdout(sys.stderr):
            print("error:", e)
            print(f"put your discord token in a file named '{token_file}'")
        sys.exit(1)


def get_channels(channels_file):
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
    help="read your discord bot token from this file",
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
    "-p",
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
def main(
    log_level,
    verbose,
    bot_token_file,
    channel,
    auth_channels_file,
    prices_file,
    fossils_file,
    users_file,
):
    auth_channels = get_channels(auth_channels_file) + list(channel)
    if not auth_channels:
        print("error: you must provide at least one authorized channel", file=sys.stderr)
        sys.exit(1)

    Turbot(
        token=get_token(bot_token_file),
        channels=auth_channels,
        prices_file=prices_file,
        fossils_file=fossils_file,
        users_file=users_file,
        log_level=getattr(logging, "DEBUG" if verbose else log_level),
    ).run()


if __name__ == "__main__":
    main()
