import inspect
import json
import random
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from os import chdir
from os.path import dirname, realpath
from pathlib import Path
from subprocess import run
from unittest.mock import MagicMock, Mock

import pytest
import pytz
from callee import Matching, String

import turbot

##############################
# Discord.py Mocks
##############################


class MockMember:
    def __init__(self, member_name, member_id, roles=[]):
        self.name = member_name
        self.id = member_id
        self.roles = roles

    def __repr__(self):
        return f"{self.name}#{self.id}"


class MockRole:
    def __init__(self, name):
        self.name = name


class MockGuild:
    def __init__(self, members):
        self.members = members


class MockChannel:
    def __init__(self, channel_type, channel_name, members):
        self.type = channel_type
        self.name = channel_name
        self.members = members
        self.guild = MockGuild(members)

        # sent is a spy for tracking calls to send(), it doesn't exist on the real object.
        # There are also helper for inspecting calls to sent defined on this class of
        # the form `last_sent_XXX` to make our lives easier.
        self.sent = MagicMock()

    async def send(self, content=None, *args, **kwargs):
        self.sent(
            content,
            **{param: value for param, value in kwargs.items() if value is not None},
        )

    @property
    def last_sent_call(self):
        args, kwargs = self.sent.call_args
        return {"args": args, "kwargs": kwargs}

    @property
    def last_sent_response(self):
        return self.last_sent_call["args"][0]

    @property
    def last_sent_embed(self):
        return self.last_sent_call["kwargs"]["embed"].to_dict()

    @property
    def all_sent_calls(self):
        sent_calls = []
        for sent_call in self.sent.call_args_list:
            args, kwargs = sent_call
            sent_calls.append({"args": args, "kwargs": kwargs})
        return sent_calls

    @property
    def all_sent_responses(self):
        return [sent_call["args"][0] for sent_call in self.all_sent_calls]

    @property
    def all_sent_embeds(self):
        return [
            sent_call["kwargs"]["embed"].to_dict()
            for sent_call in self.all_sent_calls
            if "embed" in sent_call["kwargs"]
        ]

    @property
    def all_sent_embeds_json(self):
        return json.dumps(self.all_sent_embeds, indent=4, sort_keys=True)

    @asynccontextmanager
    async def typing(self):
        yield


class MockMessage:
    def __init__(self, author, channel, content):
        self.author = author
        self.channel = channel
        self.content = content


class MockDiscordClient:
    def __init__(self):
        self.user = ADMIN


##############################
# Test Suite Constants
##############################

CLIENT_TOKEN = "my-token"
CLIENT_USER = "ADMIN"
CLIENT_USER_ID = 82226367030108160

AUTHORIZED_CHANNEL = "good channel"
UNAUTHORIZED_CHANNEL = "bad channel"

NOW = datetime(year=1982, month=4, day=24, tzinfo=pytz.utc)

TST_ROOT = dirname(realpath(__file__))
DAT_ROOT = Path(TST_ROOT) / "data"
SRC_ROOT = Path(TST_ROOT).parent

SRC_DIRS = ["tests", "turbot", "scripts"]

ADMIN_ROLE = MockRole("Turbot Admin")
PLAYER_ROLE = MockRole("ACNH Player")

ADMIN = MockMember(CLIENT_USER, CLIENT_USER_ID, roles=[ADMIN_ROLE])
FRIEND = MockMember("friend", 82169952898912256, roles=[PLAYER_ROLE])
BUDDY = MockMember("buddy", 82942320688758784, roles=[ADMIN_ROLE, PLAYER_ROLE])
GUY = MockMember("guy", 82988021019836416)
DUDE = MockMember("dude", 82988761019835305, roles=[ADMIN_ROLE])
PUNK = MockMember("punk", 119678027792646146)  # for a memeber that's not in our channel

CHANNEL_MEMBERS = [FRIEND, BUDDY, GUY, DUDE, ADMIN]

S_SPY = Mock(wraps=turbot.s)

##############################
# Test Suite Utilities
##############################


def someone():
    """Returns some non-admin user"""
    return random.choice(list(filter(lambda member: member != ADMIN, CHANNEL_MEMBERS)))


def someturbotadmin():
    """Returns a random non-admin user with the Turbot Admin role"""
    cond = lambda member: member != ADMIN and ADMIN_ROLE in member.roles
    return random.choice(list(filter(cond, CHANNEL_MEMBERS)))


def somenonturbotadmin():
    """Returns a random non-admin user without the Turbot Admin role"""
    cond = lambda member: member != ADMIN and ADMIN_ROLE not in member.roles
    return random.choice(list(filter(cond, CHANNEL_MEMBERS)))


def somebells():
    """Returns some random amount of bells"""
    return random.randint(100, 500)


def is_discord_file(obj):
    """Returns true if the given object is a discord File object."""
    return (obj.__class__.__name__) == "File"


##############################
# Test Fixtures
##############################


@pytest.fixture
def patch_discord():
    orig = turbot.Turbot.__bases__
    turbot.Turbot.__bases__ = (MockDiscordClient,)
    yield
    turbot.Turbot.__bases__ = orig


@pytest.fixture(autouse=True, scope="session")
def set_random_seed():
    random.seed(0)


@pytest.fixture
def client(monkeypatch, freezer, patch_discord, tmp_path):
    monkeypatch.setattr(turbot, "GRAPHCMD_FILE", tmp_path / "graphcmd.png")
    monkeypatch.setattr(turbot, "LASTWEEKCMD_FILE", tmp_path / "lastweek.png")
    monkeypatch.setattr(turbot, "s", S_SPY)
    freezer.move_to(NOW)
    return turbot.Turbot(
        token=CLIENT_TOKEN,
        channels=[AUTHORIZED_CHANNEL],
        prices_file=tmp_path / "prices.csv",
        art_file=tmp_path / "art.csv",
        fossils_file=tmp_path / "fossils.csv",
        users_file=tmp_path / "users.csv",
    )


@pytest.fixture
def lines():
    wrote_lines = defaultdict(int)

    def next(path):
        with open(path, "r") as f:
            rvalue = f.readlines()[wrote_lines[path] :]
            wrote_lines[path] += len(rvalue)
            return rvalue

    return next


@pytest.fixture
def graph(mocker, monkeypatch):
    def create_file(*args, **kwargs):
        Path(turbot.GRAPHCMD_FILE).touch()

    mock = mocker.Mock(side_effect=create_file)
    monkeypatch.setattr(turbot.Turbot, "generate_graph", mock)
    return mock


@pytest.fixture
def lastweek(mocker, monkeypatch):
    def create_file(*args, **kwargs):
        Path(turbot.LASTWEEKCMD_FILE).touch()

    mock = mocker.Mock(side_effect=create_file)
    monkeypatch.setattr(turbot.Turbot, "generate_graph", mock)
    return mock


@pytest.fixture
def channel():
    return MockChannel("text", AUTHORIZED_CHANNEL, members=CHANNEL_MEMBERS)


@pytest.fixture
def snap(snapshot):
    snapshot.snapshot_dir = Path("tests") / "snapshots"
    snap.counter = 0

    def match(obj):
        test = inspect.stack()[1].function
        snapshot.assert_match(str(obj), f"{test}_{snap.counter}.txt")
        snap.counter += 1

    return match


##############################
# Test Suites
##############################


@pytest.mark.asyncio
class TestTurbot:
    async def test_init(self, client):
        assert client.token == CLIENT_TOKEN
        assert client.channels == [AUTHORIZED_CHANNEL]

    async def test_on_ready(self, client):
        await client.on_ready()

    async def test_on_message_non_text(self, client, channel):
        invalid_channel_type = "voice"
        channel = MockChannel(
            invalid_channel_type, AUTHORIZED_CHANNEL, members=CHANNEL_MEMBERS
        )
        await client.on_message(MockMessage(someone(), channel, "!help"))
        channel.sent.assert_not_called()

    async def test_on_message_from_admin(self, client, channel):
        await client.on_message(MockMessage(ADMIN, channel, "!help"))
        channel.sent.assert_not_called()

    async def test_on_message_in_unauthorized_channel(self, client):
        channel = MockChannel("text", UNAUTHORIZED_CHANNEL, members=CHANNEL_MEMBERS)
        await client.on_message(MockMessage(someone(), channel, "!help"))
        channel.sent.assert_not_called()

    async def test_on_message_no_request(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!"))
        await client.on_message(MockMessage(someone(), channel, "!!"))
        await client.on_message(MockMessage(someone(), channel, "!!!"))
        await client.on_message(MockMessage(someone(), channel, "!   "))
        await client.on_message(MockMessage(someone(), channel, "!   !"))
        await client.on_message(MockMessage(someone(), channel, " !   !"))
        channel.sent.assert_not_called()

    async def test_on_message_ambiguous_request(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!h"))
        channel.sent.assert_called_with("Did you mean: !help, !hemisphere, !history?")

    async def test_on_message_invalid_request(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!xenomorph"))
        channel.sent.assert_called_with('Sorry, there is no command named "xenomorph"')

    async def test_on_message_sell_no_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell"))
        channel.sent.assert_called_with(
            "Please include selling price after command name."
        )

    async def test_on_message_sell_nonnumeric_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell foot"))
        channel.sent.assert_called_with("Selling price must be a number.")

    async def test_on_message_sell_nonpositive_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell 0"))
        channel.sent.assert_called_with("Selling price must be greater than zero.")

    async def test_on_message_sell(self, client, lines, channel):
        # initial sale
        author = someone()
        amount = somebells()
        await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
        channel.sent.assert_called_with(
            f"Logged selling price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},sell,{amount},{NOW}\n",
        ]

        # same price sale
        await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
        channel.sent.assert_called_with(
            f"Logged selling price of {amount} for user {author}. "
            f"(Same as last selling price)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{amount},{NOW}\n"]

        # higher price sale
        new_amount = amount + somebells()
        await client.on_message(MockMessage(author, channel, f"!sell {new_amount}"))
        channel.sent.assert_called_with(
            f"Logged selling price of {new_amount} for user {author}. "
            f"(Higher than last selling price of {amount} bells)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{new_amount},{NOW}\n"]

        # lower price sale
        last_amount = round(amount / 2)
        await client.on_message(MockMessage(author, channel, f"!sell {last_amount}"))
        channel.sent.assert_called_with(
            f"Logged selling price of {last_amount} for user {author}. "
            f"(Lower than last selling price of {new_amount} bells)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{last_amount},{NOW}\n"]

    async def test_on_message_buy_no_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy"))
        channel.sent.assert_called_with("Please include buying price after command name.")

    async def test_on_message_buy_nonnumeric_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy foot"))
        channel.sent.assert_called_with("Buying price must be a number.")

    async def test_on_message_buy_nonpositive_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy 0"))
        channel.sent.assert_called_with("Buying price must be greater than zero.")

    async def test_on_message_buy(self, client, lines, channel):
        author = someone()
        amount = somebells()
        await client.on_message(MockMessage(author, channel, f"!buy {amount}"))
        channel.sent.assert_called_with(
            f"Logged buying price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},buy,{amount},{NOW}\n",
        ]

    async def test_on_message_help(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!help"))
        channel.sent.assert_called_with(String())  # TODO: Verify help response?

    async def test_on_message_clear(self, client, lines, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, f"!buy {somebells()}"))
        await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))
        await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))

        await client.on_message(MockMessage(author, channel, "!clear"))
        channel.sent.assert_called_with(f"**Cleared history for {author}.**")
        assert lines(client.prices_file) == ["author,kind,price,timestamp\n"]

    async def test_on_message_bestsell(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
        await client.on_message(MockMessage(GUY, channel, "!buy 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestsell"))
        channel.sent.assert_called_with(
            "__**Best Selling Prices in the Last 12 Hours**__\n"
            f"> {BUDDY}: 600 bells at {NOW}\n"
            f"> {FRIEND}: 200 bells at {NOW}"
        )

    async def test_on_message_bestsell_timezone(self, client, channel):
        friend_tz = "America/Los_Angeles"
        await client.on_message(MockMessage(FRIEND, channel, f"!timezone {friend_tz}"))
        friend_now = NOW.astimezone(pytz.timezone("America/Los_Angeles"))

        buddy_tz = "Canada/Saskatchewan"
        await client.on_message(MockMessage(BUDDY, channel, f"!timezone {buddy_tz}"))
        buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

        guy_tz = "Africa/Abidjan"
        await client.on_message(MockMessage(GUY, channel, f"!timezone {guy_tz}"))
        # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
        await client.on_message(MockMessage(GUY, channel, "!buy 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestsell"))
        channel.sent.assert_called_with(
            "__**Best Selling Prices in the Last 12 Hours**__\n"
            f"> {BUDDY}: 600 bells at {buddy_now}\n"
            f"> {FRIEND}: 200 bells at {friend_now}"
        )

    async def test_on_message_oops(self, client, lines, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!oops"))
        channel.sent.assert_called_with(f"**Deleting last logged price for {author}.**")
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},buy,1,{NOW}\n",
            f"{author.id},sell,2,{NOW}\n",
        ]

    async def test_on_message_history_bad_name(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, f"!history {PUNK.name}"))
        channel.sent.assert_called_with(
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_command_with_blank_name(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!listfossils   "))
        channel.sent.assert_called_with("Can not find the user named  in this channel.")

    async def test_on_message_history_without_name(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!history"))
        channel.sent.assert_called_with(
            f"__**Historical info for {author}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells at {NOW}\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells at {NOW}\n"
            f"> Can buy turnips from Daisy Mae for 3 bells at {NOW}"
        )

    async def test_on_message_history_with_name(self, client, channel):
        await client.on_message(MockMessage(BUDDY, channel, "!buy 1"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 2"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 3"))

        await client.on_message(MockMessage(GUY, channel, f"!history {BUDDY.name}"))
        channel.sent.assert_called_with(
            f"__**Historical info for {BUDDY}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells at {NOW}\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells at {NOW}\n"
            f"> Can buy turnips from Daisy Mae for 3 bells at {NOW}"
        )

    async def test_on_message_history_timezone(self, client, channel):
        author = someone()
        their_tz = "America/Los_Angeles"
        await client.on_message(MockMessage(author, channel, f"!timezone {their_tz}"))
        their_now = NOW.astimezone(pytz.timezone(their_tz))

        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!history"))
        channel.sent.assert_called_with(
            f"__**Historical info for {author}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells at {their_now}\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells at {their_now}\n"
            f"> Can buy turnips from Daisy Mae for 3 bells at {their_now}"
        )

    async def test_on_message_bestbuy(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestbuy"))
        channel.sent.assert_called_with(
            "__**Best Buying Prices in the Last 12 Hours**__\n"
            f"> {BUDDY}: 60 bells at {NOW}\n"
            f"> {FRIEND}: 100 bells at {NOW}"
        )

    async def test_on_message_bestbuy_timezone(self, client, channel):
        friend_tz = "America/Los_Angeles"
        await client.on_message(MockMessage(FRIEND, channel, f"!timezone {friend_tz}"))
        friend_now = NOW.astimezone(pytz.timezone("America/Los_Angeles"))

        buddy_tz = "Canada/Saskatchewan"
        await client.on_message(MockMessage(BUDDY, channel, f"!timezone {buddy_tz}"))
        buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

        guy_tz = "Africa/Abidjan"
        await client.on_message(MockMessage(GUY, channel, f"!timezone {guy_tz}"))
        # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestbuy"))
        channel.sent.assert_called_with(
            "__**Best Buying Prices in the Last 12 Hours**__\n"
            f"> {BUDDY}: 60 bells at {buddy_now}\n"
            f"> {FRIEND}: 100 bells at {friend_now}"
        )

    async def test_on_message_turnippattern_happy_paths(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!turnippattern 100 86"))
        snap(channel.last_sent_response)

        await client.on_message(MockMessage(someone(), channel, "!turnippattern 100 99"))
        snap(channel.last_sent_response)

        await client.on_message(MockMessage(someone(), channel, "!turnippattern 100 22"))
        snap(channel.last_sent_response)

    async def test_on_message_turnippattern_invalid_params(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!turnippattern 100"))
        channel.sent.assert_called_with(
            "Please provide Daisy Mae's price and your Monday morning price\n"
            "eg. !turnippattern <buy price> <Monday morning sell price>"
        )

        await client.on_message(MockMessage(someone(), channel, "!turnippattern 1 2 3"))
        channel.sent.assert_called_with(
            "Please provide Daisy Mae's price and your Monday morning price\n"
            "eg. !turnippattern <buy price> <Monday morning sell price>"
        )

    async def test_on_message_turnippattern_nonnumeric_prices(self, client, channel):
        await client.on_message(
            MockMessage(someone(), channel, "!turnippattern something nothing")
        )
        channel.sent.assert_called_with("Prices must be numbers.")

    async def test_on_message_graph_without_user(self, client, graph, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, "!graph"))
        channel.sent.assert_called_with(
            "__**Historical Graph for All Users**__", file=Matching(is_discord_file)
        )
        graph.assert_called_with(channel, None, turbot.GRAPHCMD_FILE)
        assert Path(turbot.GRAPHCMD_FILE).exists()

    async def test_on_message_graph_with_user(self, client, graph, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, f"!graph {BUDDY.name}"))
        channel.sent.assert_called_with(
            f"__**Historical Graph for {BUDDY}**__", file=Matching(is_discord_file)
        )
        graph.assert_called_with(channel, f"{BUDDY}", turbot.GRAPHCMD_FILE)
        assert Path(turbot.GRAPHCMD_FILE).exists()

    async def test_on_message_graph_with_bad_name(self, client, graph, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, f"!graph {PUNK.name}"))
        channel.sent.assert_called_with(
            f"Can not find the user named {PUNK.name} in this channel."
        )
        graph.assert_not_called()
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    async def test_on_message_lastweek_none(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!lastweek"))
        channel.sent.assert_called_with("No graph from last week.")

    async def test_on_message_lastweek_capitalized(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!LASTWEEK"))
        channel.sent.assert_called_with("No graph from last week.")

    async def test_on_message_lastweek(self, client, freezer, lastweek, channel):
        await client.on_message(MockMessage(someturbotadmin(), channel, "!reset"))
        channel.sent.assert_called_with("**Resetting data for a new week!**")
        lastweek.assert_called_with(channel, None, turbot.LASTWEEKCMD_FILE)
        assert Path(turbot.LASTWEEKCMD_FILE).exists()

        await client.on_message(MockMessage(someone(), channel, "!lastweek"))
        channel.sent.assert_called_with(
            "__**Historical Graph from Last Week**__", file=Matching(is_discord_file)
        )

    async def test_on_message_reset_not_admin(self, client, lines, freezer, channel):
        # first log some buy and sell prices
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(FRIEND, channel, "!buy 101"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 601"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 121"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 91"))
        await client.on_message(MockMessage(GUY, channel, "!buy 100"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))
        await client.on_message(MockMessage(GUY, channel, "!buy 101"))
        await client.on_message(MockMessage(GUY, channel, "!sell 801"))

        # then jump ahead a week and log some more
        later = NOW + timedelta(days=7)
        freezer.move_to(later)
        await client.on_message(MockMessage(FRIEND, channel, "!buy 102"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 602"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 122"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 92"))
        await client.on_message(MockMessage(GUY, channel, "!buy 102"))
        await client.on_message(MockMessage(GUY, channel, "!sell 802"))

        old_data = lines(client.prices_file)

        # then reset price data
        await client.on_message(MockMessage(somenonturbotadmin(), channel, "!reset"))
        channel.sent.assert_called_with("User is not a Turbot Admin")
        with open(client.prices_file) as f:
            assert f.readlines() == old_data

        assert not Path(turbot.LASTWEEKCMD_FILE).exists()

    async def test_on_message_reset_admin(
        self, client, lines, freezer, lastweek, channel
    ):
        # first log some buy and sell prices
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(FRIEND, channel, "!buy 101"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 601"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 121"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 91"))
        await client.on_message(MockMessage(GUY, channel, "!buy 100"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))
        await client.on_message(MockMessage(GUY, channel, "!buy 101"))
        await client.on_message(MockMessage(GUY, channel, "!sell 801"))

        # then jump ahead a week and log some more
        later = NOW + timedelta(days=7)
        freezer.move_to(later)
        await client.on_message(MockMessage(FRIEND, channel, "!buy 102"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 602"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 122"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 92"))
        await client.on_message(MockMessage(GUY, channel, "!buy 102"))
        await client.on_message(MockMessage(GUY, channel, "!sell 802"))

        old_data = lines(client.prices_file)

        # then reset price data
        await client.on_message(MockMessage(someturbotadmin(), channel, "!reset"))
        channel.sent.assert_called_with("**Resetting data for a new week!**")
        with open(client.prices_file) as f:
            assert f.readlines() == [
                "author,kind,price,timestamp\n",
                f"{FRIEND.id},buy,102,{later}\n",
                f"{BUDDY.id},buy,122,{later}\n",
                f"{GUY.id},buy,102,{later}\n",
            ]
        lastweek.assert_called_with(channel, None, turbot.LASTWEEKCMD_FILE)
        assert Path(turbot.LASTWEEKCMD_FILE).exists()

        # ensure the backup is correct
        backup_file = Path(client.last_backup_filename())
        assert backup_file.exists()
        with open(backup_file) as f:
            assert old_data == f.readlines()

    async def test_on_message_collect_no_list(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!collect"))
        channel.sent.assert_called_with(
            "Please provide the name of a fossil to mark as collected."
        )

    async def test_on_message_collect(self, client, lines, channel):
        # first collect some valid fossils
        author = someone()
        fossils = "amber, ammonite  ,ankylo skull,amber, a foot"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
        channel.sent.assert_called_with(
            "Marked the following fossils as collected:\n"
            "> amber, ammonite, ankylo skull\n"
            "Did not recognize the following fossils:\n"
            "> a foot"
        )
        assert set(lines(client.fossils_file)) == {
            "author,name\n",
            f"{author.id},amber\n",
            f"{author.id},ankylo skull\n",
            f"{author.id},ammonite\n",
        }

        # collect them again
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
        channel.sent.assert_called_with(
            "The following fossils had already been collected:\n"
            "> amber, ammonite, ankylo skull\n"
            "Did not recognize the following fossils:\n"
            "> a foot"
        )

        # then collect some more with dupes
        fossils = "amber,an arm,plesio body"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
        channel.sent.assert_called_with(
            "Marked the following fossils as collected:\n"
            "> plesio body\n"
            "The following fossils had already been collected:\n"
            "> amber\n"
            "Did not recognize the following fossils:\n"
            "> an arm"
        )
        assert lines(client.fossils_file) == [f"{author.id},plesio body\n"]

    async def test_on_message_collect_congrats(self, client, channel):
        everything = sorted(list(turbot.FOSSILS))
        some, rest = everything[:10], everything[10:]

        # someone else collects some
        fossils = "amber, ammonite, ankylo skull"
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        # you collect some
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
        )

        # someone else again collects some
        fossils = "plesio body, ankylo skull"
        await client.on_message(MockMessage(FRIEND, channel, f"!collect {fossils}"))

        # then you collect all the rest
        rest_str = ", ".join(rest)
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {rest_str}"))
        channel.sent.assert_called_with(
            "Marked the following fossils as collected:\n"
            f"> {rest_str}\n"
            "**Congratulations, you've collected all fossils!**"
        )

    async def test_on_message_uncollectart_no_list(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!uncollectart"))
        snap(channel.last_sent_response)

    async def test_on_message_uncollectart_with_only_bad(self, client, channel, snap):
        art = "anime waifu, wall scroll"
        await client.on_message(MockMessage(someone(), channel, f"!uncollectart {art}"))
        snap(channel.last_sent_response)

    async def test_on_message_uncollectart(self, client, lines, channel, snap):
        # first collect some fossils
        author = someone()
        art = "great statue, sinking painting ,academic painting"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))

        # then delete some of them
        art = "great statue, anime waifu, ancient statue, academic painting"
        await client.on_message(MockMessage(author, channel, f"!uncollectart {art}"))
        snap(channel.last_sent_response)

        with open(client.art_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},sinking painting\n"]

        # and delete one more
        await client.on_message(
            MockMessage(author, channel, f"!uncollectart sinking painting")
        )
        snap(channel.last_sent_response)

        with open(client.art_file) as f:
            assert f.readlines() == ["author,name\n"]

    async def test_on_message_artsearch_no_list(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!artsearch"))
        snap(channel.last_sent_response)

    async def test_on_message_artsearch_no_need(self, client, channel, snap):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(
                GUY,
                channel,
                "!collectart sinking painting, great statue, wistful painting",
            )
        )

        await client.on_message(
            MockMessage(PUNK, channel, "!artsearch sinking painting, great statue")
        )
        snap(channel.last_sent_response)

    async def test_on_message_artsearch_no_need_with_bad(self, client, channel, snap):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(
                GUY,
                channel,
                "!collectart sinking painting, great statue, wistful painting",
            )
        )

        await client.on_message(
            MockMessage(
                PUNK, channel, "!artsearch sinking painting, great statue, anime waifu"
            )
        )
        snap(channel.last_sent_response)

    async def test_on_message_artsearch(self, client, channel, snap):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collectart sinking painting")
        )
        await client.on_message(
            MockMessage(GUY, channel, "!collectart sinking painting, great statue")
        )

        query = "sinking painting, great statue, wistful painting"
        await client.on_message(MockMessage(PUNK, channel, f"!artsearch {query}"))
        snap(channel.last_sent_response)

    async def test_on_message_artsearch_with_bad(self, client, channel, snap):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collectart sinking painting")
        )
        await client.on_message(
            MockMessage(GUY, channel, "!collectart sinking painting, great statue")
        )

        query = "sinking painting, great statue, wistful painting, anime waifu"
        await client.on_message(MockMessage(PUNK, channel, f"!artsearch {query}"))
        snap(channel.last_sent_response)

    async def test_on_message_artcount_no_params(self, client, lines, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!artcount"))
        snap(channel.last_sent_response)

    async def test_on_message_artcount_bad_name(self, client, lines, channel, snap):
        await client.on_message(MockMessage(someone(), channel, f"!artcount {PUNK.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_artcount_no_art(self, client, lines, channel, snap):
        await client.on_message(
            MockMessage(someone(), channel, f"!artcount {BUDDY.name}")
        )
        snap(channel.last_sent_response)

    async def test_on_message_artcount(self, client, lines, channel, snap):
        author = someone()
        await client.on_message(
            MockMessage(FRIEND, channel, "!collectart sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collectart sinking painting")
        )
        await client.on_message(
            MockMessage(GUY, channel, "!collectart sinking painting, great statue")
        )

        users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
        await client.on_message(MockMessage(author, channel, f"!artcount {users}"))
        snap(channel.last_sent_response)

    async def test_on_message_collectedart_no_name(self, client, lines, channel, snap):
        author = someone()
        art = "sinking painting, academic painting, great statue"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))

        await client.on_message(MockMessage(author, channel, "!collectedart"))
        snap(channel.last_sent_response)

    async def test_on_message_collectedart_with_name(self, client, lines, channel, snap):
        art = "sinking painting, academic painting, great statue"
        await client.on_message(MockMessage(GUY, channel, f"!collectart {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collectedart {GUY.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_collectedart_bad_name(self, client, lines, channel, snap):
        await client.on_message(MockMessage(BUDDY, channel, f"!collectedart {PUNK.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_collectart_no_list(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!collectart"))
        snap(channel.last_sent_response)

    async def test_on_message_collectart(self, client, lines, channel, snap):
        # first collect some art
        author = someone()
        art = "academic painting, sinking painting, anime waifu"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))
        snap(channel.last_sent_response)

        assert set(lines(client.art_file)) == {
            "author,name\n",
            f"{author.id},academic painting\n",
            f"{author.id},sinking painting\n",
        }

        # collect them again
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))
        snap(channel.last_sent_response)

        # collect some new stuff, but with some dupes
        art = "body pillow, sinking painting, tremendous statue"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))
        snap(channel.last_sent_response)

        assert lines(client.art_file) == [f"{author.id},tremendous statue\n"]

    async def test_on_message_collectart_congrats(self, client, lines, channel, snap):
        everything = sorted(list(turbot.ART.name.unique()))
        some, rest = everything[:10], everything[10:]

        # someone else collects some pieces
        art = "academic painting, sinking painting, tremendous statue"
        await client.on_message(MockMessage(GUY, channel, f"!collectart {art}"))

        # Buddy collects some
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collectart {', '.join(some)}")
        )

        # Friend collects a different set
        art = "mysterious painting, twinkling painting"
        await client.on_message(MockMessage(FRIEND, channel, f"!collectart {art}"))

        # Buddy collects the rest
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collectart {', '.join(rest)}")
        )
        snap(channel.last_sent_response)

    async def test_on_message_listart_bad_name(self, client, lines, channel, snap):
        # first collect some fossils
        author = someone()
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))

        # then list them
        await client.on_message(MockMessage(author, channel, f"!listart {PUNK.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_listart_congrats(self, client, lines, channel, snap):
        # first collect some fossils
        author = someone()
        everything = ",".join(sorted(list(turbot.ART.name.unique())))
        await client.on_message(MockMessage(author, channel, f"!collectart {everything}"))

        # then list them
        await client.on_message(MockMessage(author, channel, f"!listart"))
        snap(channel.last_sent_response)

    async def test_on_message_listart_no_name(self, client, lines, channel, snap):
        # first collect some fossils
        author = someone()
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(author, channel, f"!collectart {art}"))

        # then list them
        await client.on_message(MockMessage(author, channel, f"!listart"))
        snap(channel.last_sent_response)

    async def test_on_message_listart_with_name(self, client, lines, channel, snap):
        # first collect some fossils
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(GUY, channel, f"!collectart {art}"))

        # then list them
        await client.on_message(MockMessage(BUDDY, channel, f"!listart"))
        snap(channel.last_sent_response)

    async def test_on_message_fossilsearch_no_list(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!fossilsearch"))
        channel.sent.assert_called_with(
            "Please provide the name of a fossil to lookup users that don't have it."
        )

    async def test_on_message_fossilsearch_no_need(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
        )

        await client.on_message(
            MockMessage(PUNK, channel, "!fossilsearch amber, ammonite")
        )
        channel.sent.assert_called_with("No one currently needs this.")

    async def test_on_message_fossilsearch_no_need_with_bad(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
        )

        await client.on_message(
            MockMessage(PUNK, channel, "!fossilsearch amber, ammonite, unicorn bits")
        )
        channel.sent.assert_called_with(
            "__**Fossil Search**__\n"
            "> No one needs: amber, ammonite\n"
            "Did not recognize the following fossils:\n"
            "> unicorn bits"
        )

    async def test_on_message_fossilsearch(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        query = "amber, ammonite, ankylo skull"
        await client.on_message(MockMessage(PUNK, channel, f"!fossilsearch {query}"))
        assert channel.last_sent_response == (
            "__**Fossil Search**__\n"
            f"> {BUDDY} needs: ammonite, ankylo skull\n"
            f"> {FRIEND} needs: ankylo skull\n"
            f"> {GUY} needs: ankylo skull"
        )

    async def test_on_message_fossilsearch_with_bad(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        await client.on_message(
            MockMessage(
                PUNK, channel, "!fossilsearch amber, ammonite, ankylo skull, unicorn bits"
            )
        )
        assert channel.last_sent_response == (
            "__**Fossil Search**__\n"
            f"> {BUDDY} needs: ammonite, ankylo skull\n"
            f"> {FRIEND} needs: ankylo skull\n"
            f"> {GUY} needs: ankylo skull\n"
            "Did not recognize the following fossils:\n"
            "> unicorn bits"
        )

    async def test_on_message_fossilsearch_with_only_bad(self, client, channel):
        await client.on_message(MockMessage(PUNK, channel, "!fossilsearch unicorn bits"))
        channel.sent.assert_called_with(
            "__**Fossil Search**__\n"
            "Did not recognize the following fossils:\n"
            "> unicorn bits"
        )

    async def test_on_message_uncollect_no_list(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!uncollect"))
        channel.sent.assert_called_with(
            "Please provide the name of a fossil to mark as uncollected."
        )

    async def test_on_message_uncollect(self, client, lines, channel):
        # first collect some fossils
        author = someone()
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        # then delete some of them
        fossils = "amber, a foot, coprolite, ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!uncollect {fossils}"))
        channel.sent.assert_called_with(
            "Unmarked the following fossils as collected:\n"
            "> amber, ankylo skull\n"
            "The following fossils were already marked as not collected:\n"
            "> coprolite\n"
            "Did not recognize the following fossils:\n"
            "> a foot"
        )
        with open(client.fossils_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},ammonite\n"]

        # and delete one more
        await client.on_message(MockMessage(author, channel, f"!uncollect ammonite"))
        channel.sent.assert_called_with(
            "Unmarked the following fossils as collected:\n> ammonite"
        )
        with open(client.fossils_file) as f:
            assert f.readlines() == ["author,name\n"]

    async def test_on_message_uncollect_with_only_bad(self, client, lines, channel):
        fossils = "a foot, unicorn bits"
        await client.on_message(MockMessage(someone(), channel, f"!uncollect {fossils}"))
        channel.sent.assert_called_with(
            "Did not recognize the following fossils:\n> a foot, unicorn bits"
        )

    async def test_on_message_allfossils(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!allfossils"))
        snap(channel.last_sent_response)

    async def test_on_message_listfossils_bad_name(self, client, lines, channel):
        # first collect some fossils
        author = someone()
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        # then list them
        await client.on_message(MockMessage(author, channel, f"!listfossils {PUNK.name}"))
        channel.sent.assert_called_with(
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_listfossils_congrats(self, client, lines, channel):
        author = someone()
        everything = ", ".join(sorted(turbot.FOSSILS))
        await client.on_message(MockMessage(author, channel, f"!collect {everything}"))

        await client.on_message(MockMessage(author, channel, "!listfossils"))
        channel.sent.assert_called_with(
            "**Congratulations, you've collected all fossils!**"
        )

    async def test_on_message_listfossils_no_name(self, client, lines, channel, snap):
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(GUY, channel, "!listfossils"))
        snap(channel.last_sent_response)

    async def test_on_message_listfossils_with_name(self, client, lines, channel, snap):
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!listfossils {GUY.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_neededfossils(self, client, channel):
        everything = sorted(list(turbot.FOSSILS))

        fossils = ",".join(everything[3:])
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {fossils}"))

        fossils = ",".join(everything[20:])
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        fossils = ",".join(everything)
        await client.on_message(MockMessage(FRIEND, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(someone(), channel, "!neededfossils"))
        channel.sent.assert_called_with(
            f"> **{BUDDY}** needs acanthostega, amber, ammonite\n"
            f"> **{GUY}** needs _more than 10 fossils..._"
        )

    async def test_on_message_collectedfossils_no_name(self, client, lines, channel):
        author = someone()
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(author, channel, "!collectedfossils"))
        channel.sent.assert_called_with(
            f"__**3 Fossils donated by {author}**__\n" ">>> amber, ammonite, ankylo skull"
        )

    async def test_on_message_collectedfossils_with_name(self, client, lines, channel):
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        await client.on_message(
            MockMessage(BUDDY, channel, f"!collectedfossils {GUY.name}")
        )
        channel.sent.assert_called_with(
            f"__**3 Fossils donated by {GUY}**__\n" ">>> amber, ammonite, ankylo skull"
        )

    async def test_on_message_collectedfossils_bad_name(self, client, lines, channel):
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collectedfossils {PUNK.name}")
        )
        channel.sent.assert_called_with(
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_fossilcount_no_params(self, client, lines, channel):
        await client.on_message(MockMessage(someone(), channel, "!fossilcount"))
        channel.sent.assert_called_with(
            "Please provide at least one user name to search for a fossil count."
        )

    async def test_on_message_fossilcount_bad_name(self, client, lines, channel):
        await client.on_message(
            MockMessage(someone(), channel, f"!fossilcount {PUNK.name}")
        )
        channel.sent.assert_called_with(
            f"__**Did not recognize the following names**__\n> {PUNK.name}"
        )

    async def test_on_message_fossilcount_no_fossils(self, client, lines, channel):
        await client.on_message(
            MockMessage(someone(), channel, f"!fossilcount {BUDDY.name}")
        )
        channel.sent.assert_called_with(
            "__**Fossil Count**__\n"
            f"> **{BUDDY}** has {len(turbot.FOSSILS)} fossils remaining."
        )

    async def test_on_message_fossilcount(self, client, lines, channel):
        author = someone()
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
        await client.on_message(MockMessage(author, channel, f"!fossilcount {users}"))
        channel.sent.assert_called_with(
            "__**Fossil Count**__\n"
            f"> **{BUDDY}** has 72 fossils remaining.\n"
            f"> **{FRIEND}** has 71 fossils remaining.\n"
            f"> **{GUY}** has 71 fossils remaining.\n"
            "__**Did not recognize the following names**__\n"
            f"> {PUNK.name}"
        )

    async def test_on_message_predict_no_buy(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!predict"))
        channel.sent.assert_called_with(f"There is no recent buy price for {author}.")

    async def test_on_message_predict_bad_user(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!predict {PUNK.name}"))
        channel.sent.assert_called_with(
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_predict(self, client, freezer, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 110"))

        freezer.move_to(NOW + timedelta(days=1))
        await client.on_message(MockMessage(author, channel, "!sell 100"))
        await client.on_message(MockMessage(author, channel, "!sell 95"))

        freezer.move_to(NOW + timedelta(days=2))
        await client.on_message(MockMessage(author, channel, "!sell 90"))
        await client.on_message(MockMessage(author, channel, "!sell 85"))

        freezer.move_to(NOW + timedelta(days=4))
        await client.on_message(MockMessage(author, channel, "!sell 90"))

        freezer.move_to(NOW + timedelta(days=5))
        await client.on_message(MockMessage(author, channel, "!sell 120"))

        await client.on_message(MockMessage(author, channel, "!predict"))
        channel.sent.assert_called_with(
            f"{author}'s turnip prediction link: "
            "https://turnipprophet.io/?prices=110...100.95.90.85...90..120"
        )

    async def test_on_message_predict_with_timezone(self, client, freezer, channel):
        author = someone()
        user_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(MockMessage(author, channel, f"!timezone {user_tz.zone}"))

        # sunday morning buy
        sunday_morning = datetime(year=2020, month=4, day=21, hour=6, tzinfo=user_tz)
        freezer.move_to(sunday_morning)
        await client.on_message(MockMessage(author, channel, "!buy 110"))

        # monday morning sell
        monday_morning = sunday_morning + timedelta(days=1)
        freezer.move_to(monday_morning)
        await client.on_message(MockMessage(author, channel, "!sell 87"))

        # monday evening sell
        monday_evening = monday_morning + timedelta(hours=14)
        freezer.move_to(monday_evening)
        await client.on_message(MockMessage(author, channel, "!sell 72"))

        await client.on_message(MockMessage(author, channel, "!predict"))
        channel.sent.assert_called_with(
            f"{author}'s turnip prediction link: "
            "https://turnipprophet.io/?prices=110.87.72"
        )

    async def test_get_last_price(self, client, freezer, channel):
        # when there's no data for the user
        assert client.get_last_price(GUY) is None

        # when there's only buy data
        freezer.move_to(NOW + timedelta(days=1))
        await client.on_message(MockMessage(GUY, channel, "!buy 102"))
        assert client.get_last_price(GUY.id) is None

        # when there's sell data for someone else
        freezer.move_to(NOW + timedelta(days=2))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 102"))
        assert client.get_last_price(GUY.id) is None

        # when there's one sell price
        freezer.move_to(NOW + timedelta(days=3))
        await client.on_message(MockMessage(GUY, channel, "!sell 82"))
        assert client.get_last_price(GUY.id) == 82

        # when there's more than one sell price
        freezer.move_to(NOW + timedelta(days=4))
        await client.on_message(MockMessage(GUY, channel, "!sell 45"))
        await client.on_message(MockMessage(GUY, channel, "!sell 98"))
        assert client.get_last_price(GUY.id) == 98

    async def test_on_message_hemisphere_no_params(self, client, lines, channel):
        await client.on_message(MockMessage(someone(), channel, "!hemisphere"))
        channel.sent.assert_called_with(
            "Please provide the name of your hemisphere, northern or southern."
        )

    async def test_on_message_hemisphere_bad_hemisphere(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!hemisphere upwards"))
        channel.sent.assert_called_with(
            'Please provide either "northern" or "southern" as your hemisphere name.'
        )

    async def test_on_message_hemisphere(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere souTherN"))
        channel.sent.assert_called_with(f"Hemisphere preference registered for {author}.")
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone\n",
                f"{author.id},southern,\n",
            ]

        await client.on_message(MockMessage(author, channel, "!hemisphere NoRthErn"))
        channel.sent.assert_called_with(f"Hemisphere preference registered for {author}.")
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone\n",
                f"{author.id},northern,\n",
            ]

    async def test_on_message_fish_no_hemisphere(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!fish"))
        channel.sent.assert_called_with(
            "Please enter your hemisphere choice first using the !hemisphere command."
        )

    async def test_on_message_fish_none_found(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))

        await client.on_message(MockMessage(author, channel, "!fish Blinky"))
        channel.sent.assert_called_with('Did not find any fish searching for "Blinky".')

    async def test_on_message_fish_multiple_users(self, client, channel):
        await client.on_message(MockMessage(GUY, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(BUDDY, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(FRIEND, channel, "!hemisphere northern"))

        await client.on_message(MockMessage(GUY, channel, "!fish sea"))
        await client.on_message(MockMessage(BUDDY, channel, "!fish sea"))
        await client.on_message(MockMessage(FRIEND, channel, "!fish sea"))

    async def test_on_message_fish_search_query(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish ch"))
        snap(channel.last_sent_response)

    async def test_on_message_fish_search_leaving(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish leaving"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_fish_search_arriving(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish arriving"))
        snap(channel.last_sent_response)

    async def test_on_message_fish(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish"))
        snap(channel.last_sent_response)

    async def test_on_message_timezone_no_params(self, client, lines, channel):
        await client.on_message(MockMessage(someone(), channel, "!timezone"))
        channel.sent.assert_called_with("Please provide the name of your timezone.")

    async def test_on_message_timezone_bad_timezone(self, client, channel):
        await client.on_message(
            MockMessage(someone(), channel, "!timezone Mars/Noctis_City")
        )
        channel.sent.assert_called_with(
            "Please provide a valid timezone name, see "
            "https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for the "
            "complete list of TZ names."
        )

    async def test_on_message_timezone(self, client, channel):
        author = someone()
        await client.on_message(
            MockMessage(author, channel, "!timezone America/Los_Angeles")
        )
        channel.sent.assert_called_with(f"Timezone preference registered for {author}.")
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone\n",
                f"{author.id},,America/Los_Angeles\n",
            ]

        await client.on_message(
            MockMessage(author, channel, "!timezone Canada/Saskatchewan")
        )
        channel.sent.assert_called_with(f"Timezone preference registered for {author}.")
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone\n",
                f"{author.id},,Canada/Saskatchewan\n",
            ]

    async def test_load_prices_new(self, client):
        prices = client.load_prices()
        assert prices.empty

        loaded_dtypes = [str(t) for t in prices.dtypes.tolist()]
        assert loaded_dtypes == ["int64", "object", "int64", "datetime64[ns, UTC]"]

    async def test_load_prices_existing(self, client):
        data = [
            ["author", "kind", "price", "timestamp",],
            ["82169952898912256", "buy", "94", "2020-04-12 13:11:22.759958744+00:00"],
            ["82169952898912256", "sell", "66", "2020-04-13 12:51:41.321097374+00:00"],
            ["82169952898912256", "sell", "57", "2020-04-13 16:09:53.589281321+00:00"],
            ["82169952898912256", "sell", "130", "2020-04-14 13:04:16.417927504+00:00"],
            ["82226367030108160", "sell", "76", "2020-04-15 12:51:36.569223404+00:00"],
            ["82226367030108160", "sell", "134", "2020-04-15 16:03:58.559760571+00:00"],
            ["93126903363301376", "buy", "99", "2020-04-12 13:40:10.002708912+00:00"],
            ["93126903363301376", "sell", "87", "2020-04-13 14:25:10.902356148+00:00"],
            ["93126903363301376", "sell", "84", "2020-04-13 16:35:31.403252602+00:00"],
        ]
        with open(client.prices_file, "w") as f:
            for line in data:
                f.write(f"{','.join(line)}\n")

        prices = client.load_prices()
        loaded_data = [[str(i) for i in row.tolist()] for _, row in prices.iterrows()]
        assert loaded_data == data[1:]

        loaded_dtypes = [str(t) for t in prices.dtypes.tolist()]
        assert loaded_dtypes == ["int64", "object", "int64", "datetime64[ns, UTC]"]

    async def test_on_message_bug_no_hemisphere(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!bugs"))
        channel.sent.assert_called_with(
            "Please enter your hemisphere choice first using the !hemisphere command."
        )

    async def test_on_message_bug_none_found(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs Shelob"))
        channel.sent.assert_called_with('Did not find any bugs searching for "Shelob".')

    async def test_on_message_bug_multiple_users(self, client, channel):
        await client.on_message(MockMessage(GUY, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(BUDDY, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(FRIEND, channel, "!hemisphere northern"))

        await client.on_message(MockMessage(GUY, channel, "!bugs butt"))
        await client.on_message(MockMessage(BUDDY, channel, "!bugs butt"))
        await client.on_message(MockMessage(FRIEND, channel, "!bugs butt"))

    async def test_on_message_bug_search_query_many(
        self, client, channel, monkeypatch, snap
    ):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs butt"))
        snap(channel.last_sent_response)

    async def test_on_message_bug_search_query_few(
        self, client, channel, monkeypatch, snap
    ):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs beet"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_bug_header(self, client, channel, monkeypatch, snap):
        monkeypatch.setattr(random, "randint", lambda l, h: 100)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs butt"))
        snap(channel.last_sent_response)

    async def test_on_message_bug_search_leaving(
        self, client, channel, monkeypatch, snap
    ):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs leaving"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_bug_search_arriving(
        self, client, channel, monkeypatch, snap
    ):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs arriving"))
        snap(channel.last_sent_response)

    async def test_on_message_new(self, client, channel, monkeypatch, snap):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!new"))
        snap(channel.all_sent_responses[1])
        snap(channel.all_sent_responses[2])

    async def test_on_message_bug(self, client, channel, monkeypatch, snap):
        monkeypatch.setattr(random, "randint", lambda l, h: 0)
        author = someone()
        await client.on_message(MockMessage(author, channel, "!hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs"))
        snap(channel.all_sent_responses[1])
        snap(channel.all_sent_responses[2])

    async def test_on_message_art_fulllist(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!art"))
        snap(channel.last_sent_response)

    async def test_on_message_art_correctnames(self, client, channel, snap):
        await client.on_message(
            MockMessage(someone(), channel, "!art amazing painting, proper painting",)
        )
        snap(channel.last_sent_response)

    async def test_on_message_art_invalidnames(self, client, channel, snap):
        await client.on_message(
            MockMessage(someone(), channel, "!art academic painting, asdf",)
        )
        snap(channel.last_sent_response)

    async def test_get_graph_bad_user(self, client, channel):
        client.get_graph(channel, PUNK.name, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    async def test_get_graph_no_users(self, client, channel):
        client.get_graph(channel, None, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    async def test_get_graph_invalid_users(self, client, channel):
        with open(client.prices_file, "w") as f:
            f.writelines(
                [
                    "author,kind,price,timestamp\n",
                    f"{PUNK.id},buy,100,1982-04-24 01:00:00+00:00\n",
                ]
            )
        client.get_graph(channel, None, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    async def test_paginate(self, client):
        def subject(text):
            return [page for page in client.paginate(text)]

        assert subject("") == [""]
        assert subject("four") == ["four"]

        with open(Path(DAT_ROOT) / "ipsum_2011.txt") as f:
            text = f.read()
            pages = subject(text)
            assert len(pages) == 2
            assert all(len(page) <= 2000 for page in pages)
            assert pages == [text[0:1937], text[1937:]]

        with open(Path(DAT_ROOT) / "aaa_2001.txt") as f:
            text = f.read()
            pages = subject(text)
            assert len(pages) == 2
            assert all(len(page) <= 2000 for page in pages)
            assert pages == [text[0:2000], text[2000:]]

    async def test_humanize_months(self):
        def subject(*args):
            row = dict(
                zip(
                    [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ],
                    args,
                )
            )
            return list(turbot.humanize_months(row))

        assert subject(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) == ["the entire year"]
        assert subject(1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0) == ["Jan - Mar", "Jul - Sep"]
        assert subject(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1) == ["Jan - Mar", "Dec"]
        assert subject(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) == ["Mar"]
        assert subject(0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1) == ["Mar", "Oct - Dec"]
        assert subject(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == ["Jan"]
        assert subject(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == []


class TestFigures:
    @pytest.mark.mpl_image_compare
    def test_get_graph_all(self, client, channel):
        with open(client.prices_file, "w") as f:
            f.writelines(
                [
                    "author,kind,price,timestamp\n",
                    f"{FRIEND.id},buy,100,1982-04-24 01:00:00+00:00\n",
                    f"{FRIEND.id},sell,87,1982-04-24 01:00:00+00:00\n",
                    f"{FRIEND.id},buy,110,1982-04-24 02:00:00+00:00\n",
                    f"{FRIEND.id},sell,105,1982-04-24 02:00:00+00:00\n",
                    f"{BUDDY.id},buy,115,1982-04-24 03:00:00+00:00\n",
                    f"{BUDDY.id},sell,82,1982-04-24 03:00:00+00:00\n",
                    f"{BUDDY.id},buy,60,1982-04-24 04:00:00+00:00\n",
                    f"{BUDDY.id},sell,111,1982-04-24 04:00:00+00:00\n",
                    f"{GUY.id},buy,65,1982-04-24 05:00:00+00:00\n",
                    f"{GUY.id},sell,120,1982-04-24 05:00:00+00:00\n",
                    f"{GUY.id},buy,121,1982-04-24 06:00:00+00:00\n",
                    f"{GUY.id},sell,61,1982-04-24 06:00:00+00:00\n",
                    f"{FRIEND.id},buy,106,1982-04-24 07:00:00+00:00\n",
                    f"{FRIEND.id},sell,72,1982-04-24 07:00:00+00:00\n",
                    f"{BUDDY.id},buy,86,1982-04-24 08:00:00+00:00\n",
                    f"{BUDDY.id},sell,112,1982-04-24 08:00:00+00:00\n",
                    f"{GUY.id},buy,94,1982-04-24 09:00:00+00:00\n",
                    f"{GUY.id},sell,96,1982-04-24 09:00:00+00:00\n",
                    f"{FRIEND.id},buy,100,1982-04-26 01:00:00+00:00\n",
                    f"{FRIEND.id},sell,87,1982-04-26 01:00:00+00:00\n",
                    f"{FRIEND.id},buy,110,1982-04-26 02:00:00+00:00\n",
                    f"{FRIEND.id},sell,105,1982-04-26 02:00:00+00:00\n",
                    f"{BUDDY.id},buy,115,1982-04-26 03:00:00+00:00\n",
                    f"{BUDDY.id},sell,82,1982-04-26 03:00:00+00:00\n",
                    f"{BUDDY.id},buy,60,1982-04-26 04:00:00+00:00\n",
                    f"{BUDDY.id},sell,111,1982-04-26 04:00:00+00:00\n",
                    f"{GUY.id},buy,65,1982-04-26 05:00:00+00:00\n",
                    f"{GUY.id},sell,120,1982-04-26 05:00:00+00:00\n",
                    f"{GUY.id},buy,121,1982-04-26 06:00:00+00:00\n",
                    f"{GUY.id},sell,61,1982-04-26 06:00:00+00:00\n",
                    f"{FRIEND.id},buy,106,1982-04-26 07:00:00+00:00\n",
                    f"{FRIEND.id},sell,72,1982-04-26 07:00:00+00:00\n",
                    f"{BUDDY.id},buy,86,1982-04-26 08:00:00+00:00\n",
                    f"{BUDDY.id},sell,112,1982-04-26 08:00:00+00:00\n",
                    f"{GUY.id},buy,94,1982-04-26 09:00:00+00:00\n",
                    f"{GUY.id},sell,96,1982-04-26 09:00:00+00:00\n",
                ]
            )
        return client.get_graph(channel, None, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_single(self, client, channel):
        with open(client.prices_file, "w") as f:
            f.writelines(
                [
                    "author,kind,price,timestamp\n",
                    f"{FRIEND.id},buy,100,1982-04-24 01:00:00+00:00\n",
                    f"{FRIEND.id},sell,87,1982-04-24 01:00:00+00:00\n",
                    f"{FRIEND.id},buy,110,1982-04-24 02:00:00+00:00\n",
                    f"{FRIEND.id},sell,105,1982-04-24 02:00:00+00:00\n",
                    f"{BUDDY.id},buy,115,1982-04-24 03:00:00+00:00\n",
                    f"{BUDDY.id},sell,82,1982-04-24 03:00:00+00:00\n",
                    f"{BUDDY.id},buy,60,1982-04-24 04:00:00+00:00\n",
                    f"{BUDDY.id},sell,111,1982-04-24 04:00:00+00:00\n",
                    f"{GUY.id},buy,65,1982-04-24 05:00:00+00:00\n",
                    f"{GUY.id},sell,120,1982-04-24 05:00:00+00:00\n",
                    f"{GUY.id},buy,121,1982-04-24 06:00:00+00:00\n",
                    f"{GUY.id},sell,61,1982-04-24 06:00:00+00:00\n",
                    f"{FRIEND.id},buy,106,1982-04-24 07:00:00+00:00\n",
                    f"{FRIEND.id},sell,72,1982-04-24 07:00:00+00:00\n",
                    f"{BUDDY.id},buy,86,1982-04-24 08:00:00+00:00\n",
                    f"{BUDDY.id},sell,112,1982-04-24 08:00:00+00:00\n",
                    f"{GUY.id},buy,94,1982-04-24 09:00:00+00:00\n",
                    f"{GUY.id},sell,96,1982-04-24 09:00:00+00:00\n",
                    f"{FRIEND.id},buy,100,1982-04-26 01:00:00+00:00\n",
                    f"{FRIEND.id},sell,87,1982-04-26 01:00:00+00:00\n",
                    f"{FRIEND.id},buy,110,1982-04-26 02:00:00+00:00\n",
                    f"{FRIEND.id},sell,105,1982-04-26 02:00:00+00:00\n",
                    f"{BUDDY.id},buy,115,1982-04-26 03:00:00+00:00\n",
                    f"{BUDDY.id},sell,82,1982-04-26 03:00:00+00:00\n",
                    f"{BUDDY.id},buy,60,1982-04-26 04:00:00+00:00\n",
                    f"{BUDDY.id},sell,111,1982-04-26 04:00:00+00:00\n",
                    f"{GUY.id},buy,65,1982-04-26 05:00:00+00:00\n",
                    f"{GUY.id},sell,120,1982-04-26 05:00:00+00:00\n",
                    f"{GUY.id},buy,121,1982-04-26 06:00:00+00:00\n",
                    f"{GUY.id},sell,61,1982-04-26 06:00:00+00:00\n",
                    f"{FRIEND.id},buy,106,1982-04-26 07:00:00+00:00\n",
                    f"{FRIEND.id},sell,72,1982-04-26 07:00:00+00:00\n",
                    f"{BUDDY.id},buy,86,1982-04-26 08:00:00+00:00\n",
                    f"{BUDDY.id},sell,112,1982-04-26 08:00:00+00:00\n",
                    f"{GUY.id},buy,94,1982-04-26 09:00:00+00:00\n",
                    f"{GUY.id},sell,96,1982-04-26 09:00:00+00:00\n",
                ]
            )
        return client.get_graph(channel, FRIEND.name, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_single_no_data(self, client, channel):
        return client.get_graph(channel, FRIEND.name, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_all_no_data(self, client, channel):
        with open(client.prices_file, "w") as f:
            f.writelines(
                [
                    "author,kind,price,timestamp\n",
                    f"{FRIEND.id},buy,100,1982-04-24 01:00:00+00:00\n",
                ]
            )
        return client.get_graph(channel, None, turbot.GRAPHCMD_FILE)


class TestCodebase:
    def test_flake8(self):
        """Assures that the Python codebase passes configured Flake8 checks."""
        chdir(SRC_ROOT)
        proc = run(["flake8", *SRC_DIRS], capture_output=True)
        assert proc.returncode == 0, f"Flake8 issues:\n{proc.stdout.decode('utf-8')}"

    def test_black(self):
        """Assures that the Python codebase passes configured black checks."""
        chdir(SRC_ROOT)
        proc = run(["black", "--check", *SRC_DIRS], capture_output=True)
        assert proc.returncode == 0, f"black issues:\n{proc.stdout.decode('utf-8')}"

    def test_isort(self):
        """Assures that the Python codebase imports are correctly sorted."""
        chdir(SRC_ROOT)
        proc = run(["isort", "-df", "-rc", "-c", *SRC_DIRS], capture_output=True)
        assert proc.returncode == 0, f"isort issues:\n{proc.stdout.decode('utf-8')}"


class TestMeta:
    # This test will fail in isolation, you must run the full test suite
    # for it to actually pass. This is because it tracks the usage of
    # string keys over the entire test session. It can fail for two reasons:
    #
    # 1. There's a key in strings.yaml that's not being used at all.
    # 2. There's a key in strings.yaml that isn't being used in the tests.
    #
    # For situation #1 the solution is to remove the key from the config.
    # As for #2, there should be a new test which utilizes this key.
    def test_strings(self):
        """Assues that there are no missing or unused strings data."""
        used_keys = set(s_call[0][0] for s_call in S_SPY.call_args_list)
        config_keys = set(turbot.STRINGS.keys())
        assert config_keys - used_keys == set()
