#!/usr/bin/env python

import inspect
import logging
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from os.path import dirname, realpath
from pathlib import Path
from string import Template

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

ROOT = dirname(realpath(__file__))
PRICES_FILE = "prices.csv"
FOSSILS_FILE = "fossils.csv"
GRAPHCMD_FILE = "graphcmd.png"
LASTWEEKCMD_FILE = "lastweek.png"
PRICES_DATA = None
FOSSILS_DATA = None

with open(Path(ROOT) / "data" / "strings.yaml") as f:
    STRINGS = load(f, Loader=Loader)

with open(Path(ROOT) / "data" / "fossils.txt") as f:
    FOSSILS = frozenset([line.strip().lower() for line in f.readlines()])


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


def build_prices():
    """Returns an empty DataFrame suitable for storing price data."""
    return pd.DataFrame(columns=["author", "kind", "price", "timestamp"]).astype(
        {"timestamp": "datetime64[ns, UTC]"}
    )


def save_prices(data):
    """Saves the given prices data to csv file."""
    global PRICES_DATA
    PRICES_DATA = data
    PRICES_DATA.to_csv(PRICES_FILE, index=False)


def load_prices():
    """Returns a DataFrame of price data or creates an empty one if there isn't any."""
    global PRICES_DATA
    if PRICES_DATA is None:
        try:
            PRICES_DATA = pd.read_csv(PRICES_FILE).astype(
                {"timestamp": "datetime64[ns, UTC]"}
            )
        except FileNotFoundError:
            PRICES_DATA = build_prices()
    return PRICES_DATA


def save_fossils(data):
    """Saves the given fossils data to csv file."""
    global FOSSILS_DATA
    FOSSILS_DATA = data
    FOSSILS_DATA.to_csv(FOSSILS_FILE, index=False)


def build_fossils():
    """Returns an empty DataFrame suitable for storing fossil data."""
    return pd.DataFrame(columns=["author", "name"])


def load_fossils():
    """Returns a DataFrame of fossils data or creates an empty one if there isn't any."""
    global FOSSILS_DATA
    if FOSSILS_DATA is None:
        try:
            FOSSILS_DATA = pd.read_csv(FOSSILS_FILE)
        except FileNotFoundError:
            FOSSILS_DATA = build_fossils()
    return FOSSILS_DATA


def generate_graph(channel, user, graphname):
    HOURS = mdates.HourLocator()
    HOURS_FMT = mdates.DateFormatter("%b %d %H:%M")
    TWELVEHOUR = mdates.HourLocator(interval=12)

    plt.figure(figsize=(10, 12), dpi=100)
    _, ax = plt.subplots()
    ax.xaxis.set_major_locator(TWELVEHOUR)
    ax.xaxis.set_major_formatter(HOURS_FMT)
    ax.xaxis.set_minor_locator(HOURS)

    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))

    priceList = load_prices()
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
    plt.xlabel("Time (Eastern)")
    plt.title("Selling Prices")
    plt.legend(legendElems, loc="upper left", bbox_to_anchor=(1, 1))

    figure = plt.gcf()
    figure.set_size_inches(18, 9)

    plt.savefig(graphname, dpi=100)
    plt.close("all")

    return True


def append_price(author, kind, price):
    """Adds a price to the prices data file for the given author and kind."""
    prices = load_prices()
    prices = prices.append(
        pd.DataFrame(
            columns=prices.columns,
            data=[[author.id, kind, price, datetime.now(pytz.utc)]],
        ),
        ignore_index=True,
    )
    save_prices(prices)


def get_last_price(user_id):
    """Returns the last sell price for the given user id."""
    prices = load_prices()
    last = prices[prices.author == user_id].sort_values(by=["timestamp"]).tail(1).price
    return int(last) if last.any() else None


def paginate(text):
    """Discord responses must be 2000 characters of less; paginate can break them up."""
    breakpoints = ["\n", ".", ",", "-", " "]
    remaining = text
    while len(remaining) > 2000:
        breakpoint = 1999

        for char in breakpoints:
            index = remaining.rfind(char, 1800, 1999)
            if index != -1:
                breakpoint = index
                break

        yield remaining[0:breakpoint]
        remaining = remaining[breakpoint + 1 :]
    yield remaining


class Turbot(discord.Client):
    """Discord turnip bot"""

    def __init__(self, token, channels):
        super().__init__()
        self.token = token
        self.channels = channels

    def run(self):
        super().run(self.token)

    async def process(self, message):
        tokens = message.content.split(" ")
        request, params = tokens[0].lstrip("!"), tokens[1:]
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
            pages = list(paginate(response))
            last_page_index = len(pages) - 1
            for i, page in enumerate(pages):
                file = attachment if attachment and i == last_page_index else None
                await message.channel.send(page, file=file)

    async def on_message(self, message):
        if (
            str(message.channel.type) == "text"
            and message.author.id != self.user.id
            and message.channel.name in self.channels
            and message.content.startswith("!")
        ):
            await self.process(message)

    async def on_ready(self):
        logging.debug("logged in as %s", self.user)

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

            title = f"!{command.__name__.replace('_command', '')}"
            if params:
                title = f"{title} {params}"
            usage += f"\n> **{title}**"
            usage += f"\n>    {use}"
            usage += "\n> "
        usage += "\n> turbot created by TheAstropath"
        return usage, None

    def hello_command(self, channel, author, params):
        """
        Says hello to you.
        """
        return s("hello"), None

    def lookup_command(self, channel, author, params):
        """
        Gets a user's id given a name. | <name>
        """
        if not params:
            return s("lookup_no_params"), None
        name = " ".join(params)
        user_id = discord_user_id(channel, name)
        return s("lookup", user_id=user_id), None

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

        last_price = get_last_price(author.id)

        logging.debug("saving sell price of %s bells for user id %s", price, author.id)
        append_price(author, "sell", price)

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
        append_price(author, "buy", price)
        return s("buy", price=price, name=author), None

    def reset_command(self, channel, author, params):
        """
        DO NOT USE UNLESS ASKED. Generates a final graph for use with !lastweek and
        resets all data for all users.
        """
        generate_graph(channel, None, LASTWEEKCMD_FILE)
        prices = load_prices()
        buys = prices[prices.kind == "buy"].sort_values(by="timestamp")
        idx = buys.groupby(by="author")["timestamp"].idxmax()
        prices = buys.loc[idx]
        save_prices(prices)
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
            generate_graph(channel, None, GRAPHCMD_FILE)
            return s("graph_all_users"), discord.File(GRAPHCMD_FILE)

        user_id = discord_user_id(channel, params[0])
        user_name = discord_user_name(channel, user_id)
        if not user_id or not user_name:
            return s("user_cant_find", name=params[0]), None

        generate_graph(channel, user_name, GRAPHCMD_FILE)
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
            return s("user_cant_find", name=target), None

        prices = load_prices()
        yours = prices[prices.author == target_id]
        lines = [s("history_header", name=target_name)]
        for _, row in yours.iterrows():
            lines.append(
                s(f"history_{row.kind}", price=row.price, timestamp=row.timestamp)
            )
        return "\n".join(lines), None

    def oops_command(self, channel, author, params):
        """
        Remove your last logged turnip price.
        """
        target = author.id
        target_name = discord_user_name(channel, target)
        prices = load_prices()
        prices = prices.drop(prices[prices.author == author.id].tail(1).index)
        save_prices(prices)
        return s("oops", name=target_name), None

    def clear_command(self, channel, author, params):
        """
        Clears all of your own historical turnip prices.
        """
        user_id = discord_user_id(channel, str(author))
        prices = load_prices()
        prices = prices[prices.author != user_id]
        save_prices(prices)
        return s("clear", name=author), None

    def _best(self, channel, author, kind):
        prices = load_prices()
        past = datetime.now(pytz.utc) - timedelta(hours=12)
        sells = prices[(prices.kind == kind) & (prices.timestamp > past)]
        idx = sells.groupby(by="author").price.transform(max) == sells.price
        bests = sells[idx].sort_values(by="price", ascending=kind == "buy")
        lines = [s(f"best{kind}_header")]
        for _, row in bests.iterrows():
            name = discord_user_from_id(channel, row.author)
            lines.append(s("best", name=name, price=row.price, timestamp=row.timestamp))
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

        fossils = load_fossils()
        yours = fossils[fossils.author == author.id]
        dupes = yours.loc[yours.name.isin(valid)].name.values.tolist()
        new_names = list(set(valid) - set(dupes))
        new_data = [[author.id, name] for name in new_names]
        new_fossils = pd.DataFrame(columns=fossils.columns, data=new_data)
        fossils = fossils.append(new_fossils, ignore_index=True)
        save_fossils(fossils)

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

        fossils = load_fossils()
        yours = fossils[fossils.author == author.id]
        previously_collected = yours.loc[yours.name.isin(valid)]
        deleted = set(previously_collected.name.values.tolist())
        didnt_have = valid - deleted
        fossils = fossils.drop(previously_collected.index)
        save_fossils(fossils)

        lines = []
        if deleted:
            lines.append(s("uncollect_deleted", items=", ".join(sorted(deleted))))
        if didnt_have:
            lines.append(s("uncollect_already", items=", ".join(sorted(didnt_have))))
        if invalid:
            lines.append(s("uncollect_bad", items=", ".join(sorted(invalid))))
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

        fossils = load_fossils()
        theirs = fossils[fossils.author != author.id]
        them = theirs.author.unique()
        results = defaultdict(list)
        for fossil in valid:
            havers = theirs[theirs.name == fossil].author.unique()
            needers = np.setdiff1d(them, havers).tolist()
            for needer in needers:
                name = discord_user_from_id(channel, needer)
                results[name].append(fossil)

        if not results:
            return s("fossilsearch_noneed"), None

        lines = [s("fossilsearch_header")]
        for name, needed in results.items():
            need_list = fossils = ", ".join(sorted(needed))
            lines.append(s("fossilsearch_row", name=name, fossils=need_list))
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
            return s("user_cant_find", name=target), None

        fossils = load_fossils()
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
            return s("user_cant_find", name=target), None

        fossils = load_fossils()
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
            fossils = load_fossils()
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


def get_token():
    """Returns the discord token from token.txt."""
    try:
        with open("token.txt", "r") as token_file:
            return token_file.readline().strip()
    except IOError as e:
        with redirect_stdout(sys.stderr):
            print("error:", e)
            print("put your discord token in a file named 'token.txt'")
        sys.exit(1)


def get_channels():
    """Returns the authorized channels from channels.txt."""
    try:
        with open("channels.txt", "r") as channels_file:
            return [line.strip() for line in channels_file.readlines()]
    except IOError as e:
        with redirect_stdout(sys.stderr):
            print("error:", e)
            print("put your authorized channels in a file named 'channels.txt'")
        sys.exit(1)


def main():
    logging.basicConfig(level=logging.DEBUG)
    Turbot(get_token(), get_channels()).run()


if __name__ == "__main__":
    main()
