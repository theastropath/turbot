import inspect
import json
import random
import re
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
import toml
from callee import Matching

import turbot
from turbot.assets import load_strings

##############################
# Discord.py Mocks
##############################


class MockFile:
    def __init__(self, fp):
        self.fp = fp


class MockMember:
    def __init__(self, member_name, member_id, roles=[]):
        self.name = member_name
        self.id = member_id
        self.roles = roles
        self.avatar_url = "http://example.com/avatar.png"

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
        # There are also helpers for inspecting calls to sent defined on this class of
        # the form `last_sent_XXX` and `all_sent_XXX` to make our lives easier.
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
    def all_sent_files(self):
        return [
            sent_call["kwargs"]["file"]
            for sent_call in self.all_sent_calls
            if "file" in sent_call["kwargs"]
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
REPO_ROOT = Path(TST_ROOT).parent
SRC_ROOT = REPO_ROOT / "src"

SRC_DIRS = [REPO_ROOT / "tests", SRC_ROOT / "turbot", REPO_ROOT / "scripts"]

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
def client(monkeypatch, mocker, freezer, patch_discord, tmp_path):
    monkeypatch.setattr(turbot, "GRAPHCMD_FILE", tmp_path / "graphcmd.png")
    monkeypatch.setattr(turbot, "LASTWEEKCMD_FILE", tmp_path / "lastweek.png")
    monkeypatch.setattr(turbot, "s", S_SPY)

    # To ensure fast tests, assume that calls to generate_graph() always fails. To test
    # that they should succeed, use the fixture graph() or lastweek(). To actually test
    # the generation of a graph, put the test into TestFigures and use client.get_graph()
    # instead of client.generate_graph().
    monkeypatch.setattr(turbot.Turbot, "generate_graph", mocker.Mock(return_value=False))

    freezer.move_to(NOW)
    return turbot.Turbot(
        token=CLIENT_TOKEN, channels=[AUTHORIZED_CHANNEL], db_dir=tmp_path,
    )


@pytest.fixture
def graph(mocker, monkeypatch):
    def create_file(*args, **kwargs):
        Path(turbot.GRAPHCMD_FILE).touch()
        return True

    mock = mocker.Mock(side_effect=create_file, return_value=True)
    monkeypatch.setattr(turbot.Turbot, "generate_graph", mock)
    return mock


@pytest.fixture
def lastweek(mocker, monkeypatch):
    def create_file(*args, **kwargs):
        Path(turbot.LASTWEEKCMD_FILE).touch()
        return True

    mock = mocker.Mock(side_effect=create_file, return_value=True)
    monkeypatch.setattr(turbot.Turbot, "generate_graph", mock)
    return mock


@pytest.fixture
def channel():
    return MockChannel("text", AUTHORIZED_CHANNEL, members=CHANNEL_MEMBERS)


SNAPSHOTS_USED = set()


@pytest.fixture
def snap(snapshot):
    snapshot.snapshot_dir = Path("tests") / "snapshots"
    snap.counter = 0

    def match(obj):
        test = inspect.stack()[1].function
        snapshot_file = f"{test}_{snap.counter}.txt"
        snapshot.assert_match(str(obj), snapshot_file)
        snap.counter += 1
        SNAPSHOTS_USED.add(snapshot_file)

    return match


@pytest.fixture
def with_bugs_header(monkeypatch):
    monkeypatch.setattr(random, "randint", lambda l, h: 100)  # 100% chance of bugs header


@pytest.fixture
def without_bugs_header(monkeypatch):
    monkeypatch.setattr(random, "randint", lambda l, h: 0)  # 0% chance of bugs header


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
        assert channel.last_sent_response == ("Did you mean: !help, !history?")

    async def test_on_message_invalid_request(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!xenomorph"))
        assert channel.last_sent_response == (
            'Sorry, there is no command named "xenomorph"'
        )

    async def test_process_long_response_with_file(self, client, channel, monkeypatch):
        file = MockFile("file")

        @turbot.command
        def mock_help(channel, author, params):
            return "What? " * 1000, file

        monkeypatch.setattr(client, "help", mock_help)
        await client.on_message(MockMessage(someone(), channel, "!help"))
        assert len(channel.all_sent_responses) == 3
        assert channel.all_sent_files == [file]

    async def test_process_weird_response(self, client, channel, monkeypatch):
        @turbot.command
        def mock_help(channel, author, params):
            return 42, None  # can't send int as a response

        monkeypatch.setattr(client, "help", mock_help)
        with pytest.raises(RuntimeError):
            await client.on_message(MockMessage(someone(), channel, "!help"))

    async def test_on_message_sell_at_time_with_tz(self, client, channel, freezer):
        author = someone()
        author_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
        )

        monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
        monday_evening = monday_morning + timedelta(hours=13)
        # user's time is 8 hours ahead of utc on this date:
        monday_evening_adjust = monday_evening + timedelta(hours=8)
        command_time = monday_morning + timedelta(days=3)
        freezer.move_to(command_time)

        amount = somebells()
        await client.on_message(
            MockMessage(author, channel, f"!sell {amount} monday evening")
        )
        assert channel.last_sent_response == (
            f"Logged selling price of {amount} for user {author}."
        )
        assert client.data.prices.values.tolist() == [
            [author.id, "sell", amount, monday_evening_adjust],
        ]

    # async def test_get_user_timeline(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     sunday_am = datetime(2020, 4, 26, 9, tzinfo=author_tz)
    #     freezer.move_to(sunday_am)
    #     assert sunday_am.isoweekday() == turbot.DAYS["sunday"]
    #     await client.on_message(MockMessage(author, channel, "!buy 90"))

    #     amount = 100
    #     monday_am = sunday_am + timedelta(days=1)
    #     for offset in range(0, 50):
    #         freezer.move_to(monday_am + timedelta(hours=offset))
    #         await client.on_message(
    #             MockMessage(author, channel, f"!sell {amount} monday morning")
    #         )
    #         amount += 5

    #     assert client.get_user_timeline(author.id) == [
    #         90,
    #         345,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #     ]

    # async def test_get_user_timeline_buy_monday(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     sunday_am = datetime(2020, 4, 27, 9, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am)
    #     await client.on_message(MockMessage(author, channel, "!buy 90"))

    #     amount = 100
    #     monday_am = sunday_am + timedelta(days=1)
    #     for offset in range(0, 50):
    #         freezer.move_to(monday_am + timedelta(hours=offset))
    #         await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
    #         amount += 5

    #     assert client.get_user_timeline(author.id) == [
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #     ]

    # async def test_get_user_timeline_buy_at_sunday(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     sunday_am = datetime(2020, 5, 4, 6, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am + timedelta(days=2, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!buy 90 sunday morning"))

    #     await client.on_message(MockMessage(author, channel, "!history"))
    #     assert channel.last_sent_response == (
    #         f"__**Historical info for {author}**__\n"
    #         "> Can buy turnips from Daisy Mae for 90 bells 3 days ago (Sunday am)"
    #     )

    # async def test_get_user_timeline_no_sells(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am)
    #     await client.on_message(MockMessage(author, channel, "!buy 90"))

    #     assert client.get_user_timeline(author.id) == [
    #         90,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #     ]

    # async def test_get_user_timeline_sunday_sells(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am)
    #     await client.on_message(MockMessage(author, channel, "!buy 90"))

    #     amount = 100
    #     monday_am = sunday_am + timedelta(days=1)
    #     for offset in range(0, 50):
    #         freezer.move_to(monday_am + timedelta(hours=offset))
    #         await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
    #         amount += 5

    #     await client.on_message(MockMessage(author, channel, "!sell 50 sunday evening"))

    #     assert client.get_user_timeline(author.id) == [
    #         90,
    #         145,
    #         205,
    #         265,
    #         325,
    #         345,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #     ]

    # async def test_on_message_sell_at_time(self, client, channel, lines, freezer):
    #     monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
    #     monday_evening = monday_morning + timedelta(hours=13)
    #     command_time = monday_morning + timedelta(days=3)
    #     freezer.move_to(command_time)

    #     author = someone()
    #     amount = somebells()
    #     await client.on_message(
    #         MockMessage(author, channel, f"!sell {amount} monday evening")
    #     )
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},sell,{amount},{monday_evening}\n",
    #     ]

    # async def test_on_message_sell_bad_time(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell 100 funday"))
    #     assert channel.last_sent_response == (
    #         "Please provide both the day of the week and time of day."
    #     )

    # async def test_on_message_sell_bad_day(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell 100 fun morning"))
    #     assert channel.last_sent_response == (
    #         "Please use monday, wednesday, tuesday, etc for the day parameter."
    #     )

    # async def test_on_message_sell_incomplete_time(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell 100 friday pants"))
    #     assert channel.last_sent_response == (
    #         "Please use either morning or evening as the time parameter."
    #     )

    # async def test_on_message_sell_no_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell"))
    #     assert channel.last_sent_response == (
    #         "Please include selling price after command name."
    #     )

    # async def test_on_message_sell_nonnumeric_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell foot"))
    #     assert channel.last_sent_response == ("Selling price must be a number.")

    # async def test_on_message_sell_nonpositive_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!sell 0"))
    #     assert channel.last_sent_response == ("Selling price must be greater than zero.")

    # async def test_on_message_sell_extra_space(self, client, channel, lines):
    #     author = someone()
    #     amount = somebells()
    #     await client.on_message(MockMessage(author, channel, f"!sell  {amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},sell,{amount},{NOW}\n",
    #     ]

    # async def test_on_message_sell(self, client, channel, lines):
    #     # initial sale
    #     author = someone()
    #     amount = somebells()
    #     await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},sell,{amount},{NOW}\n",
    #     ]

    #     # same price sale
    #     await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {amount} for user {author}. "
    #         f"(Same as last selling price)"
    #     )
    #     assert lines(client.data.file("prices")) == [f"{author.id},sell,{amount},{NOW}\n"]

    #     # higher price sale
    #     new_amount = amount + somebells()
    #     await client.on_message(MockMessage(author, channel, f"!sell {new_amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {new_amount} for user {author}. "
    #         f"(Higher than last selling price of {amount} bells)"
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         f"{author.id},sell,{new_amount},{NOW}\n"
    #     ]

    #     # lower price sale
    #     last_amount = round(amount / 2)
    #     await client.on_message(MockMessage(author, channel, f"!sell {last_amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged selling price of {last_amount} for user {author}. "
    #         f"(Lower than last selling price of {new_amount} bells)"
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         f"{author.id},sell,{last_amount},{NOW}\n"
    #     ]

    # async def test_on_message_buy_at_time_with_tz(self, client, channel, lines, freezer):
    #     author = someone()
    #     author_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
    #     )

    #     monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
    #     monday_evening = monday_morning + timedelta(hours=13)
    #     # user's time is 8 hours ahead of utc on this date:
    #     monday_evening_adjust = monday_evening + timedelta(hours=8)
    #     command_time = monday_morning + timedelta(days=3)
    #     freezer.move_to(command_time)

    #     amount = somebells()
    #     await client.on_message(
    #         MockMessage(author, channel, f"!buy {amount} monday evening")
    #     )
    #     assert channel.last_sent_response == (
    #         f"Logged buying price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},buy,{amount},{monday_evening_adjust}\n",
    #     ]

    # async def test_on_message_buy_at_time(self, client, channel, lines, freezer):
    #     monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
    #     monday_evening = monday_morning + timedelta(hours=13)
    #     command_time = monday_morning + timedelta(days=3)
    #     freezer.move_to(command_time)

    #     author = someone()
    #     amount = somebells()
    #     await client.on_message(
    #         MockMessage(author, channel, f"!buy {amount} monday evening")
    #     )
    #     assert channel.last_sent_response == (
    #         f"Logged buying price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},buy,{amount},{monday_evening}\n",
    #     ]

    # async def test_on_message_buy_bad_time(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy 100 funday"))
    #     assert channel.last_sent_response == (
    #         "Please provide both the day of the week and time of day."
    #     )

    # async def test_on_message_buy_bad_day(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy 100 fun morning"))
    #     assert channel.last_sent_response == (
    #         "Please use monday, wednesday, tuesday, etc for the day parameter."
    #     )

    # async def test_on_message_buy_incomplete_time(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy 100 friday pants"))
    #     assert channel.last_sent_response == (
    #         "Please use either morning or evening as the time parameter."
    #     )

    # async def test_on_message_buy_no_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy"))
    #     assert channel.last_sent_response == (
    #         "Please include buying price after command name."
    #     )

    # async def test_on_message_buy_nonnumeric_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy foot"))
    #     assert channel.last_sent_response == ("Buying price must be a number.")

    # async def test_on_message_buy_nonpositive_price(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!buy 0"))
    #     assert channel.last_sent_response == ("Buying price must be greater than zero.")

    # async def test_on_message_buy(self, client, channel, lines):
    #     author = someone()
    #     amount = somebells()
    #     await client.on_message(MockMessage(author, channel, f"!buy {amount}"))
    #     assert channel.last_sent_response == (
    #         f"Logged buying price of {amount} for user {author}."
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},buy,{amount},{NOW}\n",
    #     ]

    # async def test_on_message_help(self, client, channel, snap):
    #     await client.on_message(MockMessage(someone(), channel, "!help"))
    #     for response in channel.all_sent_responses:
    #         snap(response)
    #     assert len(channel.all_sent_calls) == 2

    # async def test_on_message_clear(self, client, channel, lines):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, f"!buy {somebells()}"))
    #     await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))
    #     await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))

    #     await client.on_message(MockMessage(author, channel, "!clear"))
    #     assert channel.last_sent_response == (f"**Cleared history for {author}.**")
    #     assert lines(client.data.file("prices")) == ["author,kind,price,timestamp\n"]

    # async def test_on_message_best_bad_param(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!best dog"))
    #     assert channel.last_sent_response == (
    #         "Please choose either best buy or best sell."
    #     )

    # async def test_on_message_best_sell(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 800"))

    #     await client.on_message(MockMessage(someone(), channel, "!best sell"))
    #     assert channel.last_sent_response == (
    #         "__**Best Selling Prices in the Last 12 Hours**__\n"
    #         f"> **{BUDDY}:** now for 600 bells\n"
    #         f"> **{FRIEND}:** now for 200 bells"
    #     )

    # async def test_on_message_best_sell_timezone(self, client, channel):
    #     friend_tz = "America/Los_Angeles"
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, f"!pref timezone {friend_tz}")
    #     )
    #     friend_now = NOW.astimezone(pytz.timezone(friend_tz))

    #     buddy_tz = "Canada/Saskatchewan"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!pref timezone {buddy_tz}"))
    #     buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

    #     guy_tz = "Africa/Abidjan"
    #     await client.on_message(MockMessage(GUY, channel, f"!pref timezone {guy_tz}"))
    #     # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 800"))

    #     await client.on_message(MockMessage(someone(), channel, "!best"))
    #     assert channel.last_sent_response == (
    #         "__**Best Selling Prices in the Last 12 Hours**__\n"
    #         f"> **{BUDDY}:** {turbot.h(buddy_now)} for 600 bells\n"
    #         f"> **{FRIEND}:** {turbot.h(friend_now)} for 200 bells"
    #     )

    # async def test_on_message_oops(self, client, channel, lines):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!buy 1"))
    #     await client.on_message(MockMessage(author, channel, "!sell 2"))
    #     await client.on_message(MockMessage(author, channel, "!buy 3"))

    #     await client.on_message(MockMessage(author, channel, "!oops"))
    #     assert channel.last_sent_response == (
    #         f"**Deleting last logged price for {author}.**"
    #     )
    #     assert lines(client.data.file("prices")) == [
    #         "author,kind,price,timestamp\n",
    #         f"{author.id},buy,1,{NOW}\n",
    #         f"{author.id},sell,2,{NOW}\n",
    #     ]

    # async def test_on_message_history_bad_name(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!buy 1"))
    #     await client.on_message(MockMessage(author, channel, "!sell 2"))
    #     await client.on_message(MockMessage(author, channel, "!buy 3"))

    #     await client.on_message(MockMessage(author, channel, f"!history {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_history_without_name(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!buy 1"))
    #     await client.on_message(MockMessage(author, channel, "!sell 2"))
    #     await client.on_message(MockMessage(author, channel, "!buy 3"))

    #     await client.on_message(MockMessage(author, channel, "!history"))
    #     ts = f"{turbot.h(NOW)} ({turbot.day_and_time(NOW)})"
    #     assert channel.last_sent_response == (
    #         f"__**Historical info for {author}**__\n"
    #         f"> Can buy turnips from Daisy Mae for 1 bells {ts}\n"
    #         f"> Can sell turnips to Timmy & Tommy for 2 bells {ts}\n"
    #         f"> Can buy turnips from Daisy Mae for 3 bells {ts}"
    #     )

    # async def test_on_message_history_with_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 1"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 2"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 3"))

    #     await client.on_message(MockMessage(GUY, channel, f"!history {BUDDY.name}"))
    #     ts = f"{turbot.h(NOW)} ({turbot.day_and_time(NOW)})"
    #     assert channel.last_sent_response == (
    #         f"__**Historical info for {BUDDY}**__\n"
    #         f"> Can buy turnips from Daisy Mae for 1 bells {ts}\n"
    #         f"> Can sell turnips to Timmy & Tommy for 2 bells {ts}\n"
    #         f"> Can buy turnips from Daisy Mae for 3 bells {ts}"
    #     )

    # async def test_on_message_history_timezone(self, client, channel):
    #     author = someone()
    #     their_tz = "America/Los_Angeles"
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {their_tz}")
    #     )
    #     their_now = NOW.astimezone(pytz.timezone(their_tz))

    #     await client.on_message(MockMessage(author, channel, "!buy 1"))
    #     await client.on_message(MockMessage(author, channel, "!sell 2"))
    #     await client.on_message(MockMessage(author, channel, "!buy 3"))

    #     await client.on_message(MockMessage(author, channel, "!history"))
    #     ts = f"{turbot.h(their_now)} ({turbot.day_and_time(their_now)})"
    #     assert channel.last_sent_response == (
    #         f"__**Historical info for {author}**__\n"
    #         f"> Can buy turnips from Daisy Mae for 1 bells {ts}\n"
    #         f"> Can sell turnips to Timmy & Tommy for 2 bells {ts}\n"
    #         f"> Can buy turnips from Daisy Mae for 3 bells {ts}"
    #     )

    # async def test_on_message_best_buy(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 800"))

    #     await client.on_message(MockMessage(someone(), channel, "!best buy"))
    #     assert channel.last_sent_response == (
    #         "__**Best Buying Prices in the Last 12 Hours**__\n"
    #         f"> **{BUDDY}:** now for 60 bells\n"
    #         f"> **{FRIEND}:** now for 100 bells"
    #     )

    # async def test_on_message_best_buy_timezone(self, client, channel):
    #     friend_tz = "America/Los_Angeles"
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, f"!pref timezone {friend_tz}")
    #     )
    #     friend_now = NOW.astimezone(pytz.timezone(friend_tz))

    #     buddy_tz = "Canada/Saskatchewan"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!pref timezone {buddy_tz}"))
    #     buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

    #     guy_tz = "Africa/Abidjan"
    #     await client.on_message(MockMessage(GUY, channel, f"!pref timezone {guy_tz}"))
    #     # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 800"))

    #     await client.on_message(MockMessage(someone(), channel, "!best buy"))
    #     assert channel.last_sent_response == (
    #         "__**Best Buying Prices in the Last 12 Hours**__\n"
    #         f"> **{BUDDY}:** {turbot.h(buddy_now)} for 60 bells\n"
    #         f"> **{FRIEND}:** {turbot.h(friend_now)} for 100 bells"
    #     )

    # async def test_on_message_graph(self, client, channel, graph):
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 800"))

    #     await client.on_message(MockMessage(someone(), channel, "!graph"))
    #     channel.sent.assert_called_with(
    #         "__**Historical Graph for All Users**__", file=Matching(is_discord_file)
    #     )
    #     graph.assert_called_with(channel, None, turbot.GRAPHCMD_FILE)
    #     assert Path(turbot.GRAPHCMD_FILE).exists()

    # async def test_on_message_lastweek_none(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!lastweek"))
    #     assert channel.last_sent_response == ("No graph from last week.")

    # async def test_on_message_lastweek_capitalized(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!LASTWEEK"))
    #     assert channel.last_sent_response == ("No graph from last week.")

    # async def test_on_message_lastweek(self, client, channel, lastweek):
    #     await client.on_message(MockMessage(someturbotadmin(), channel, "!reset"))
    #     assert channel.last_sent_response == ("**Resetting data for a new week!**")
    #     lastweek.assert_called_with(channel, None, turbot.LASTWEEKCMD_FILE)
    #     assert Path(turbot.LASTWEEKCMD_FILE).exists()

    #     await client.on_message(MockMessage(someone(), channel, "!lastweek"))
    #     channel.sent.assert_called_with(
    #         "__**Historical Graph from Last Week**__", file=Matching(is_discord_file)
    #     )

    # async def test_on_message_reset_not_admin(self, client, channel, lines, freezer):
    #     # first log some buy and sell prices
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 101"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 601"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 121"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 91"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 100"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 800"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 101"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 801"))

    #     # then jump ahead a week and log some more
    #     later = NOW + timedelta(days=7)
    #     freezer.move_to(later)
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 102"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 602"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 122"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 92"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 102"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 802"))

    #     old_data = lines(client.data.file("prices"))

    #     # then reset price data
    #     await client.on_message(MockMessage(somenonturbotadmin(), channel, "!reset"))
    #     assert channel.last_sent_response == ("User is not a Turbot Admin")
    #     with open(client.data.file("prices")) as f:
    #         assert f.readlines() == old_data

    #     assert not Path(turbot.LASTWEEKCMD_FILE).exists()

    # async def test_on_message_reset_admin(
    #     self, client, channel, lines, freezer, lastweek
    # ):
    #     # first log some buy and sell prices
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 101"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 601"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 121"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 91"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 100"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 800"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 101"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 801"))

    #     # then jump ahead a week and log some more
    #     later = NOW + timedelta(days=7)
    #     freezer.move_to(later)
    #     await client.on_message(MockMessage(FRIEND, channel, "!buy 102"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!sell 602"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!buy 122"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 92"))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 102"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 802"))

    #     old_data = lines(client.data.file("prices"))

    #     # then reset price data
    #     await client.on_message(MockMessage(someturbotadmin(), channel, "!reset"))
    #     assert channel.last_sent_response == ("**Resetting data for a new week!**")
    #     with open(client.data.file("prices")) as f:
    #         assert f.readlines() == [
    #             "author,kind,price,timestamp\n",
    #             f"{FRIEND.id},buy,102,{later}\n",
    #             f"{BUDDY.id},buy,122,{later}\n",
    #             f"{GUY.id},buy,102,{later}\n",
    #         ]
    #     lastweek.assert_called_with(channel, None, turbot.LASTWEEKCMD_FILE)
    #     assert Path(turbot.LASTWEEKCMD_FILE).exists()

    #     # ensure the backup is correct
    #     backup_file = Path(client.last_backup_filename())
    #     assert backup_file.exists()
    #     with open(backup_file) as f:
    #         assert old_data == f.readlines()

    # async def test_on_message_collect_no_list(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!collect"))
    #     assert channel.last_sent_response == (
    #         "Please provide the name of something to mark as collected."
    #     )

    # async def test_on_message_collect_fossils(self, client, channel, lines):
    #     # first collect some valid fossils
    #     author = someone()
    #     fossils = "amber, ammonite  ,ankylo skull,amber, a foot"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following fossils as collected:\n"
    #         "> amber, ammonite, ankylo skull\n"
    #         "Unrecognized collectable names:\n"
    #         "> a foot"
    #     )
    #     assert set(lines(client.data.file("fossils"))) == {
    #         "author,name\n",
    #         f"{author.id},amber\n",
    #         f"{author.id},ankylo skull\n",
    #         f"{author.id},ammonite\n",
    #     }

    #     # collect them again
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "The following fossils had already been collected:\n"
    #         "> amber, ammonite, ankylo skull\n"
    #         "Unrecognized collectable names:\n"
    #         "> a foot"
    #     )

    #     # then collect some more with dupes
    #     fossils = "amber,an arm,plesio body"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following fossils as collected:\n"
    #         "> plesio body\n"
    #         "The following fossils had already been collected:\n"
    #         "> amber\n"
    #         "Unrecognized collectable names:\n"
    #         "> an arm"
    #     )
    #     assert lines(client.data.file("fossils")) == [f"{author.id},plesio body\n"]

    # async def test_on_message_collect_songs_unicode(self, client, channel, lines):
    #     await client.on_message(MockMessage(someone(), channel, "!collect Café K.K."))
    #     assert channel.last_sent_response == (
    #         "Marked the following songs as collected:\n> Café K.K."
    #     )

    # async def test_on_message_collect_songs_fuzzy(self, client, channel, lines):
    #     await client.on_message(MockMessage(someone(), channel, "!collect cafe"))
    #     assert channel.last_sent_response == (
    #         "Marked the following songs as collected:\n> Café K.K."
    #     )

    # async def test_on_message_collect_fossils_congrats(self, client, channel):
    #     everything = sorted(list(client.assets["fossils"].all))
    #     some, rest = everything[:10], everything[10:]

    #     # someone else collects some
    #     fossils = "amber, ammonite, ankylo skull"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

    #     # you collect some
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
    #     )

    #     # someone else again collects some
    #     fossils = "plesio body, ankylo skull"
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {fossils}"))

    #     # then you collect all the rest
    #     rest_str = ", ".join(rest)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {rest_str}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following fossils as collected:\n"
    #         f"> {rest_str}\n"
    #         "**Congratulations, you've collected all fossils!**"
    #     )

    # async def test_on_message_uncollect_art(self, client, channel, lines):
    #     # first collect some fossils
    #     author = someone()
    #     art = "great statue, sinking painting ,academic painting"
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))

    #     # then delete some of them
    #     art = "great statue, anime waifu, ancient statue, academic painting"
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {art}"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following pieces of art as collected:\n"
    #         "> academic painting, great statue\n"
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> ancient statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("art")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},sinking painting\n"]

    #     # then delete the same ones again
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {art}"))
    #     assert channel.last_sent_response == (
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> academic painting, ancient statue, great statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("art")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},sinking painting\n"]

    #     # and delete one more
    #     await client.on_message(
    #         MockMessage(author, channel, "!uncollect sinking painting")
    #     )
    #     assert channel.last_sent_response == (
    #         "Unmarked the following pieces of art as collected:\n" "> sinking painting"
    #     )
    #     with open(client.data.file("art")) as f:
    #         assert f.readlines() == ["author,name\n"]

    # async def test_on_message_search_art_no_need_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
    #     )
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, "!collect sinking painting, great statue")
    #     )
    #     await client.on_message(
    #         MockMessage(
    #             GUY, channel, "!collect sinking painting, great statue, wistful painting",
    #         )
    #     )

    #     await client.on_message(
    #         MockMessage(
    #             PUNK, channel, "!search sinking painting, great statue, anime waifu"
    #         )
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: great statue, sinking painting\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_search_art(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect sinking painting, great statue")
    #     )

    #     query = "sinking painting, great statue, wistful painting"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     channel.last_sent_response == (
    #         "__**Art Search**__\n"
    #         f"> {BUDDY} needs: great statue, wistful painting\n"
    #         f"> {FRIEND} needs: wistful painting\n"
    #         f"> {GUY} needs: wistful painting"
    #     )

    # async def test_on_message_search_art_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect sinking painting, great statue")
    #     )

    #     query = "sinking painting, great statue, wistful painting, anime waifu"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     assert channel.last_sent_response == (
    #         "> No one needs: sinking painting\n"
    #         f"> {BUDDY} needs arts: great statue, wistful painting\n"
    #         f"> {FRIEND} needs arts: wistful painting\n"
    #         f"> {GUY} needs arts: wistful painting\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_count_no_params(self, client, channel, snap):
    #     await client.on_message(MockMessage(BUDDY, channel, "!count"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 1

    # async def test_on_message_count_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, f"!count {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**Did not recognize the following names**__\n> {PUNK.name}"
    #     )

    # async def test_on_message_count_when_nothing_collected(self, client, channel, snap):
    #     await client.on_message(MockMessage(someone(), channel, f"!count {BUDDY.name}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 1

    # async def test_on_message_count_art(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect sinking painting, great statue")
    #     )

    #     users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
    #     await client.on_message(MockMessage(author, channel, f"!count {users}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_some(self, client, channel):
    #     author = someone()
    #     art = "sinking painting, academic painting, great statue"
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))

    #     fossils = "amber, ammonite, ankylo skull"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

    #     fish = "goldfish, killifish, snapping turtle"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))

    #     bugs = "grasshopper, honeybee, robust cicada"
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 fossils donated by {author}**__\n"
    #         "> amber, ammonite, ankylo skull\n"
    #         f"__**3 bugs donated by {author}**__\n"
    #         "> grasshopper, honeybee, robust cicada\n"
    #         f"__**3 fish donated by {author}**__\n"
    #         "> goldfish, killifish, snapping turtle\n"
    #         f"__**3 pieces of art donated by {author}**__\n"
    #         "> academic painting, great statue, sinking painting"
    #     )

    # async def test_on_message_collected_all(self, client, channel):
    #     author = someone()

    #     for kind in client.assets.collectables:
    #         items = ",".join(client.assets[kind].all)
    #         await client.on_message(MockMessage(author, channel, f"!collect {items}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     for kind in client.assets.collectables:
    #         assert (
    #             f"**Congratulations, you've collected all {kind}!**"
    #             in channel.last_sent_response
    #         )

    # async def test_on_message_collected_art_no_name(self, client, channel):
    #     author = DUDE
    #     art = "sinking painting, academic painting, great statue"
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 pieces of art donated by {DUDE}**__\n"
    #         "> academic painting, great statue, sinking painting"
    #     )

    # async def test_on_message_collected_art_congrats(self, client, channel):
    #     everything = ",".join(client.assets["art"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!collected"))
    #     channel.last_sent_response == "**Congratulations, you've collected all art!**"

    # async def test_on_message_uncollected_art_congrats(self, client, channel, snap):
    #     everything = ",".join(client.assets["art"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!uncollected"))
    #     for response in channel.all_sent_responses:
    #         snap(response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_art_with_name(self, client, channel):
    #     art = "sinking painting, academic painting, great statue"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**3 pieces of art donated by {GUY}**__\n"
    #         "> academic painting, great statue, sinking painting"
    #     )

    # async def test_on_message_collected_art_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_collect_art(self, client, channel, lines):
    #     # first collect some art
    #     author = BUDDY
    #     art = "academic painting, sinking painting, anime waifu"
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following art as collected:\n"
    #         "> academic painting, sinking painting\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     assert set(lines(client.data.file("art"))) == {
    #         "author,name\n",
    #         f"{author.id},academic painting\n",
    #         f"{author.id},sinking painting\n",
    #     }

    #     # collect them again
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))
    #     assert channel.last_sent_response == (
    #         "The following art had already been collected:\n"
    #         "> academic painting, sinking painting\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )

    #     # collect some new stuff, but with some dupes
    #     art = "body pillow, sinking painting, tremendous statue"
    #     await client.on_message(MockMessage(author, channel, f"!collect {art}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following art as collected:\n"
    #         "> tremendous statue\n"
    #         "The following art had already been collected:\n"
    #         "> sinking painting\n"
    #         "Unrecognized collectable names:\n"
    #         "> body pillow"
    #     )

    #     assert lines(client.data.file("art")) == [f"{author.id},tremendous statue\n"]

    # async def test_on_message_collect_art_congrats(self, client, channel, snap):
    #     everything = sorted(list(client.assets["art"].all))
    #     some, rest = everything[:10], everything[10:]

    #     # someone else collects some pieces
    #     art = "academic painting, sinking painting, tremendous statue"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {art}"))

    #     # Buddy collects some
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
    #     )

    #     # Friend collects a different set
    #     art = "mysterious painting, twinkling painting"
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {art}"))

    #     # Buddy collects the rest
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(rest)}")
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_bad_name(self, client, channel):
    #     art = "academic painting, sinking painting"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         "Can not find the user named punk in this channel."
    #     )

    # async def test_on_message_collected_no_name(self, client, channel):
    #     art = "academic painting, sinking painting"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**2 pieces of art donated by {BUDDY}**__\n"
    #         "> academic painting, sinking painting"
    #     )

    # async def test_on_message_uncollected_bad_name(self, client, channel):
    #     art = "academic painting, sinking painting"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!uncollected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         "Can not find the user named punk in this channel."
    #     )

    # async def test_on_message_uncollected_no_name(self, client, channel, snap):
    #     art = "academic painting, sinking painting"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!uncollected"))
    #     assert channel.all_sent_responses[0] == (
    #         "Marked the following art as collected:\n"
    #         "> academic painting, sinking painting"
    #     )
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_uncollected_with_name(self, client, channel, snap):
    #     art = "academic painting, sinking painting"
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!uncollected {BUDDY.name}"))
    #     assert channel.all_sent_responses[0] == (
    #         "Marked the following art as collected:\n"
    #         "> academic painting, sinking painting"
    #     )
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_search_no_list(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!search"))
    #     assert channel.last_sent_response == (
    #         "Please provide the name of a collectable to search for."
    #     )

    # async def test_on_message_search_all_no_need(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
    #     )

    #     await client.on_message(MockMessage(PUNK, channel, "!search amber, ammonite"))
    #     assert channel.last_sent_response == ("No one currently needs this.")

    # async def test_on_message_search_fossil_no_need_with_bad(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
    #     )

    #     await client.on_message(
    #         MockMessage(PUNK, channel, "!search amber, ammonite, unicorn bits")
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: amber, ammonite\n"
    #         "Did not recognize the following collectables:\n"
    #         "> unicorn bits"
    #     )

    # async def test_on_message_search_fossil(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
    #     await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

    #     query = "amber, ammonite, ankylo skull"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     assert channel.last_sent_response == (
    #         "> No one needs: amber\n"
    #         f"> {BUDDY} needs fossils: ammonite, ankylo skull\n"
    #         f"> {FRIEND} needs fossils: ankylo skull\n"
    #         f"> {GUY} needs fossils: ankylo skull"
    #     )

    # async def test_on_message_search_fossil_with_bad(self, client, channel):
    #     await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
    #     await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

    #     await client.on_message(
    #         MockMessage(
    #             PUNK, channel, "!search amber, ammonite, ankylo skull, unicorn bits"
    #         )
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: amber\n"
    #         f"> {BUDDY} needs fossils: ammonite, ankylo skull\n"
    #         f"> {FRIEND} needs fossils: ankylo skull\n"
    #         f"> {GUY} needs fossils: ankylo skull\n"
    #         "Did not recognize the following collectables:\n"
    #         "> unicorn bits"
    #     )

    # async def test_on_message_search_with_only_bad(self, client, channel):
    #     await client.on_message(MockMessage(PUNK, channel, "!search unicorn bits"))
    #     assert channel.last_sent_response == (
    #         "Did not recognize the following collectables:\n" "> unicorn bits"
    #     )

    # async def test_on_message_uncollect_no_list(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!uncollect"))
    #     assert channel.last_sent_response == (
    #         "Please provide the name of something to mark as uncollected."
    #     )

    # async def test_on_message_uncollect_fossil(self, client, channel):
    #     # first collect some fossils
    #     author = someone()
    #     fossils = "amber, ammonite ,ankylo skull"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

    #     # then delete some of them
    #     fossils = "amber, a foot, coprolite, ankylo skull"
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following fossils as collected:\n"
    #         "> amber, ankylo skull\n"
    #         "The following fossils were already marked as not collected:\n"
    #         "> coprolite\n"
    #         "Unrecognized collectable names:\n"
    #         "> a foot"
    #     )
    #     with open(client.data.file("fossils")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},ammonite\n"]

    #     # delete the same ones again
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "The following fossils were already marked as not collected:\n"
    #         "> amber, ankylo skull, coprolite\n"
    #         "Unrecognized collectable names:\n"
    #         "> a foot"
    #     )
    #     with open(client.data.file("fossils")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},ammonite\n"]

    #     # and delete one more
    #     await client.on_message(MockMessage(author, channel, "!uncollect ammonite"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following fossils as collected:\n> ammonite"
    #     )
    #     with open(client.data.file("fossils")) as f:
    #         assert f.readlines() == ["author,name\n"]

    # async def test_on_message_uncollect_with_only_bad(self, client, channel):
    #     fossils = "a foot, unicorn bits"
    #     await client.on_message(MockMessage(someone(), channel, f"!uncollect {fossils}"))
    #     assert channel.last_sent_response == (
    #         "Unrecognized collectable names:\n> a foot, unicorn bits"
    #     )

    # async def test_on_message_collected_fossils_congrats(self, client, channel):
    #     author = someone()
    #     everything = ", ".join(sorted(client.assets["fossils"].all))
    #     await client.on_message(MockMessage(author, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         "**Congratulations, you've collected all fossils!**"
    #     )

    # async def test_on_message_uncollected_fossils_congrats(self, client, channel, snap):
    #     author = DUDE
    #     everything = ", ".join(sorted(client.assets["fossils"].all))
    #     await client.on_message(MockMessage(author, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(author, channel, "!uncollected"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_needed_no_param(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed"))
    #     assert channel.last_sent_response == (
    #         "Please provide a parameter: fossils, bugs, fish, art, or songs."
    #     )

    # async def test_on_message_needed_bad_param(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed food"))
    #     assert channel.last_sent_response == (
    #         "Invalid parameter, use one of: fossils, bugs, fish, art, or songs."
    #     )

    # async def test_on_message_needed_fossils(self, client, channel):
    #     everything = sorted(list(client.assets["fossils"].all))

    #     fossils = ",".join(everything[3:])
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {fossils}"))

    #     fossils = ",".join(everything[20:])
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

    #     fossils = ",".join(everything)
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {fossils}"))

    #     await client.on_message(MockMessage(someone(), channel, "!needed fossils"))
    #     assert channel.last_sent_response == (
    #         f"> **{BUDDY}** needs acanthostega, amber, ammonite\n"
    #         f"> **{GUY}** needs _more than 10 fossils..._"
    #     )

    # async def test_on_message_needed_songs(self, client, channel):
    #     everything = sorted(list(client.assets["songs"].all))

    #     songs = ",".join(everything[3:])
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {songs}"))

    #     songs = ",".join(everything[20:])
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {songs}"))

    #     songs = ",".join(everything)
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {songs}"))

    #     await client.on_message(MockMessage(someone(), channel, "!needed songs"))
    #     assert channel.last_sent_response == (
    #         f"> **{BUDDY}** needs Agent K.K., Aloha K.K., Animal City\n"
    #         f"> **{GUY}** needs _more than 10 songs..._"
    #     )

    # async def test_on_message_needed_fossils_none(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed fossils"))
    #     assert channel.last_sent_response == (
    #         "No fossils are known to be needed at this time, "
    #         "new users must collect at least one before being considered for this search."
    #     )

    # async def test_on_message_needed_bugs(self, client, channel):
    #     everything = sorted(list(client.assets["bugs"].all))

    #     bugs = ",".join(everything[3:])
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {bugs}"))

    #     bugs = ",".join(everything[20:])
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {bugs}"))

    #     bugs = ",".join(everything)
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {bugs}"))

    #     await client.on_message(MockMessage(someone(), channel, "!needed bugs"))
    #     assert channel.last_sent_response == (
    #         f"> **{BUDDY}** needs agrias butterfly, ant, atlas moth\n"
    #         f"> **{GUY}** needs _more than 10 bugs..._"
    #     )

    # async def test_on_message_needed_bugs_none(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed bugs"))
    #     assert channel.last_sent_response == (
    #         "No bugs are known to be needed at this time, "
    #         "new users must collect at least one before being considered for this search."
    #     )

    # async def test_on_message_needed_fish(self, client, channel):
    #     everything = sorted(list(client.assets["fish"].all))

    #     fish = ",".join(everything[3:])
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {fish}"))

    #     fish = ",".join(everything[20:])
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fish}"))

    #     fish = ",".join(everything)
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {fish}"))

    #     await client.on_message(MockMessage(someone(), channel, "!needed fish"))
    #     assert channel.last_sent_response == (
    #         f"> **{BUDDY}** needs anchovy, angelfish, arapaima\n"
    #         f"> **{GUY}** needs _more than 10 fish..._"
    #     )

    # async def test_on_message_needed_fish_none(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed fish"))
    #     assert channel.last_sent_response == (
    #         "No fish are known to be needed at this time, "
    #         "new users must collect at least one before being considered for this search."
    #     )

    # async def test_on_message_needed_art(self, client, channel):
    #     everything = sorted(list(client.assets["art"].all))

    #     art = ",".join(everything[3:])
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

    #     art = ",".join(everything[20:])
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {art}"))

    #     art = ",".join(everything)
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {art}"))

    #     await client.on_message(MockMessage(someone(), channel, "!needed art"))
    #     assert channel.last_sent_response == (
    #         f"> **{BUDDY}** needs academic painting, amazing painting, ancient statue\n"
    #         f"> **{GUY}** needs _more than 10 art..._"
    #     )

    # async def test_on_message_needed_art_none(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!needed art"))
    #     assert channel.last_sent_response == (
    #         "No art are known to be needed at this time, "
    #         "new users must collect at least one before being considered for this search."
    #     )

    # async def test_on_message_collected_fossils_no_name(self, client, channel):
    #     author = someone()
    #     fossils = "amber, ammonite ,ankylo skull"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 fossils donated by {author}**__\n" "> amber, ammonite, ankylo skull"
    #     )

    # async def test_on_message_collected_fossils_with_name(self, client, channel):
    #     fossils = "amber, ammonite ,ankylo skull"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**3 fossils donated by {GUY}**__\n" "> amber, ammonite, ankylo skull"
    #     )

    # async def test_on_message_collected_fossils_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_count_fossils(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
    #     await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

    #     users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
    #     await client.on_message(MockMessage(author, channel, f"!count {users}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_predict_no_buy(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!predict"))
    #     assert channel.last_sent_response == (
    #         f"There is no recent buy price for {author}."
    #     )

    # async def test_on_message_predict_bad_user(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, f"!predict {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_predict(self, client, channel, freezer, graph):
    #     author = someone()

    #     sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am)
    #     await client.on_message(MockMessage(author, channel, "!buy 110"))

    #     freezer.move_to(sunday_am + timedelta(days=1))
    #     await client.on_message(MockMessage(author, channel, "!sell 100"))
    #     freezer.move_to(sunday_am + timedelta(days=1, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!sell 95"))

    #     freezer.move_to(sunday_am + timedelta(days=2))
    #     await client.on_message(MockMessage(author, channel, "!sell 90"))
    #     freezer.move_to(sunday_am + timedelta(days=2, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!sell 85"))

    #     freezer.move_to(sunday_am + timedelta(days=4))
    #     await client.on_message(MockMessage(author, channel, "!sell 90"))

    #     freezer.move_to(sunday_am + timedelta(days=5))
    #     await client.on_message(MockMessage(author, channel, "!sell 120"))

    #     await client.on_message(MockMessage(author, channel, "!predict"))
    #     channel.sent.assert_called_with(
    #         f"__**Predictive Graph for {author}**__\n"
    #         "Details: <https://turnipprophet.io/?prices=110.100.95.90.85...90..120>",
    #         file=Matching(is_discord_file),
    #     )

    # async def test_on_message_predict_error(self, client, channel, freezer):
    #     author = someone()

    #     sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
    #     freezer.move_to(sunday_am)
    #     await client.on_message(MockMessage(author, channel, "!buy 102"))

    #     freezer.move_to(sunday_am + timedelta(days=1))
    #     await client.on_message(MockMessage(author, channel, "!sell 93"))
    #     freezer.move_to(sunday_am + timedelta(days=1, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!sell 87"))

    #     freezer.move_to(sunday_am + timedelta(days=2))
    #     await client.on_message(MockMessage(author, channel, "!sell 86"))
    #     freezer.move_to(sunday_am + timedelta(days=2, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!sell 79"))

    #     freezer.move_to(sunday_am + timedelta(days=3, hours=12))
    #     await client.on_message(MockMessage(author, channel, "!sell 69"))

    #     await client.on_message(MockMessage(author, channel, "!predict"))
    #     channel.sent.assert_called_with(
    #         f"__**Predictive Graph for {author}**__\n"
    #         "Details: <https://turnipprophet.io/?prices=102.93.87.86.79..69>"
    #     )

    # async def test_on_message_predict_with_timezone(
    #     self, client, channel, freezer, graph
    # ):
    #     author = someone()
    #     user_tz = pytz.timezone("America/Los_Angeles")
    #     await client.on_message(
    #         MockMessage(author, channel, f"!pref timezone {user_tz.zone}")
    #     )

    #     # sunday morning buy
    #     sunday_morning = datetime(year=2020, month=4, day=19, hour=6, tzinfo=user_tz)
    #     freezer.move_to(sunday_morning)
    #     await client.on_message(MockMessage(author, channel, "!buy 110"))

    #     # monday morning sell
    #     monday_morning = sunday_morning + timedelta(days=1)
    #     freezer.move_to(monday_morning)
    #     await client.on_message(MockMessage(author, channel, "!sell 87"))

    #     # monday evening sell
    #     monday_evening = monday_morning + timedelta(hours=14)
    #     freezer.move_to(monday_evening)
    #     await client.on_message(MockMessage(author, channel, "!sell 72"))

    #     await client.on_message(MockMessage(author, channel, "!predict"))
    #     channel.sent.assert_called_with(
    #         f"__**Predictive Graph for {author}**__\n"
    #         "Details: <https://turnipprophet.io/?prices=110.87.72>",
    #         file=Matching(is_discord_file),
    #     )

    # async def test_get_last_price(self, client, channel, freezer):
    #     # when there's no data for the user
    #     assert client.get_last_price(GUY) is None

    #     # when there's only buy data
    #     freezer.move_to(NOW + timedelta(days=1))
    #     await client.on_message(MockMessage(GUY, channel, "!buy 102"))
    #     assert client.get_last_price(GUY.id) is None

    #     # when there's sell data for someone else
    #     freezer.move_to(NOW + timedelta(days=2))
    #     await client.on_message(MockMessage(BUDDY, channel, "!sell 102"))
    #     assert client.get_last_price(GUY.id) is None

    #     # when there's one sell price
    #     freezer.move_to(NOW + timedelta(days=3))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 82"))
    #     assert client.get_last_price(GUY.id) == 82

    #     # when there's more than one sell price
    #     freezer.move_to(NOW + timedelta(days=4))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 45"))
    #     await client.on_message(MockMessage(GUY, channel, "!sell 98"))
    #     assert client.get_last_price(GUY.id) == 98

    # async def test_on_message_fish_no_hemisphere(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!fish"))
    #     assert channel.last_sent_response == (
    #         "Please enter your hemisphere choice first "
    #         "using the !pref hemisphere command."
    #     )

    # async def test_on_message_fish_none_found(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))

    #     await client.on_message(MockMessage(author, channel, "!fish Blinky"))
    #     assert channel.last_sent_response == (
    #         'Did not find any fish searching for "Blinky".'
    #     )

    # async def test_on_message_fish_multiple_users(self, client, channel):
    #     await client.on_message(MockMessage(GUY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!pref hemisphere northern"))

    #     await client.on_message(MockMessage(GUY, channel, "!fish sea"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!fish sea"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!fish sea"))

    # async def test_on_message_fish_search_query(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!fish ch"))
    #     assert channel.all_sent_responses == [
    #         f"Registered hemisphere preference for {author}.",
    #         "__**Fish available right now**__",
    #         None,
    #         None,
    #         None,
    #         "__**Fish available this month**__\n"
    #         "> **Anchovy** (shadow size 2) is available 4 am - 9 pm at sea "
    #         "(sells for 200 bells) \n"
    #         "> **Pale chub** (shadow size 1) is available 9 am - 4 pm at river "
    #         "(sells for 200 bells) \n"
    #         "> **Ranchu goldfish** (shadow size 2) is available 9 am - 4 pm at pond "
    #         "(sells for 4500 bells) ",
    #     ]
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 6

    # async def test_on_message_fish_search_leaving(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!fish leaving"))
    #     assert channel.all_sent_responses == [
    #         f"Registered hemisphere preference for {author}.",
    #         "__**Fish available right now**__",
    #         None,
    #         None,
    #         None,
    #     ]
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 5

    # async def test_on_message_fish_search_arriving(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!fish arriving"))
    #     assert channel.all_sent_responses[0] == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     assert len(channel.all_sent_calls) == 3

    # async def test_on_message_fish_few(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     most_fish = client.assets["fish"].all - {"snapping turtle", "guppy"}
    #     await client.on_message(
    #         MockMessage(author, channel, f"!collect {','.join(most_fish)}")
    #     )
    #     await client.on_message(MockMessage(author, channel, "!fish"))
    #     assert channel.all_sent_responses[0] == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     snap(channel.all_sent_responses[1])
    #     assert channel.all_sent_responses[2] == "__**Fish available right now**__"
    #     assert channel.all_sent_responses[3] is None
    #     assert channel.all_sent_responses[4] == (
    #         "__**Fish available this month**__\n"
    #         "> **Guppy** (shadow size 1) is available 9 am - 4 pm at river "
    #         "(sells for 1300 bells) _New this month_"
    #     )
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 5

    # async def test_on_message_fish(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!fish"))
    #     assert channel.all_sent_responses[0] == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_fish_case_insensitive(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!fish SeA"))
    #     assert channel.all_sent_responses == [
    #         f"Registered hemisphere preference for {author}.",
    #         "__**Fish available right now**__",
    #         None,
    #         None,
    #     ]
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_bugs_case_insensitive(
    #     self, client, channel, snap, without_bugs_header
    # ):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs TaRaNtUlA"))
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 3

    # async def test_load_prices_new(self, client):
    #     prices = client.data.prices
    #     assert prices.empty

    #     loaded_dtypes = [str(t) for t in prices.dtypes.tolist()]
    #     assert loaded_dtypes == ["int64", "object", "int64", "datetime64[ns, UTC]"]

    # async def test_load_prices_existing(self, client):
    #     data = [
    #         ["author", "kind", "price", "timestamp",],
    #         ["82169952898912256", "buy", "94", "2020-04-12 13:11:22.759958744+00:00"],
    #         ["82169952898912256", "sell", "66", "2020-04-13 12:51:41.321097374+00:00"],
    #         ["82169952898912256", "sell", "57", "2020-04-13 16:09:53.589281321+00:00"],
    #         ["82169952898912256", "sell", "130", "2020-04-14 13:04:16.417927504+00:00"],
    #         ["82226367030108160", "sell", "76", "2020-04-15 12:51:36.569223404+00:00"],
    #         ["82226367030108160", "sell", "134", "2020-04-15 16:03:58.559760571+00:00"],
    #         ["93126903363301376", "buy", "99", "2020-04-12 13:40:10.002708912+00:00"],
    #         ["93126903363301376", "sell", "87", "2020-04-13 14:25:10.902356148+00:00"],
    #         ["93126903363301376", "sell", "84", "2020-04-13 16:35:31.403252602+00:00"],
    #     ]
    #     with open(client.data.file("prices"), "w") as f:
    #         for line in data:
    #             f.write(f"{','.join(line)}\n")

    #     prices = client.data.prices
    #     loaded_data = [[str(i) for i in row.tolist()] for _, row in prices.iterrows()]
    #     assert loaded_data == data[1:]

    #     loaded_dtypes = [str(t) for t in prices.dtypes.tolist()]
    #     assert loaded_dtypes == ["int64", "object", "int64", "datetime64[ns, UTC]"]

    # async def test_on_message_bug_no_hemisphere(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!bugs"))
    #     assert channel.last_sent_response == (
    #         "Please enter your hemisphere choice "
    #         "first using the !pref hemisphere command."
    #     )

    # async def test_on_message_bug_none_found(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs Shelob"))
    #     assert channel.last_sent_response == (
    #         'Did not find any bugs searching for "Shelob".'
    #     )

    # async def test_on_message_bug_multiple_users(self, client, channel):
    #     await client.on_message(MockMessage(GUY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!pref hemisphere northern"))

    #     await client.on_message(MockMessage(GUY, channel, "!bugs butt"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!bugs butt"))
    #     await client.on_message(MockMessage(FRIEND, channel, "!bugs butt"))

    # async def test_on_message_bug_search_query_many(
    #     self, client, channel, without_bugs_header, snap
    # ):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs butt"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 2

    # async def test_on_message_bug_search_query_few(
    #     self, client, channel, without_bugs_header, snap
    # ):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs beet"))
    #     assert channel.all_sent_responses == [
    #         f"Registered hemisphere preference for {author}.",
    #         "__**Bugs available right now**__",
    #         None,
    #         None,
    #         None,
    #     ]
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 5

    # async def test_on_message_bug_header(self, client, channel, with_bugs_header, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs butt"))
    #     assert channel.all_sent_responses[0] == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     assert channel.all_sent_responses[1] == (
    #         "```diff\n"
    #         "-Eeek! What wretched things. Alas, I am obliged to respond...\n"
    #         "```"
    #     )
    #     snap(channel.all_sent_responses[2])
    #     assert len(channel.all_sent_calls) == 3

    # async def test_on_message_bug_search_leaving(
    #     self, client, channel, without_bugs_header, freezer, snap
    # ):
    #     author = someone()
    #     freezer.move_to(datetime(2020, 5, 6, 11, 30))
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs leaving"))
    #     assert channel.all_sent_responses[0] == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     assert channel.all_sent_responses[1] == "__**Bugs available right now**__"
    #     assert channel.all_sent_responses[2] is None
    #     snap(channel.all_sent_embeds_json)
    #     len(channel.all_sent_calls) == 3

    # async def test_on_message_bug_search_arriving(
    #     self, client, channel, without_bugs_header, snap
    # ):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs arriving"))
    #     assert channel.all_sent_responses[1] == "__**Bugs available right now**__"
    #     assert channel.all_sent_responses[2] is None
    #     assert channel.all_sent_responses[3] is None
    #     assert channel.all_sent_responses[4] is None
    #     assert channel.all_sent_responses[5] is None
    #     snap(channel.all_sent_responses[6])
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == 7

    # async def test_on_message_new(self, client, channel, without_bugs_header, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!new"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     snap(channel.all_sent_responses[4])
    #     assert len(channel.all_sent_calls) == 5

    # async def test_on_message_new_first_day(
    #     self, client, channel, freezer, without_bugs_header, snap
    # ):
    #     first_day_of_the_month = datetime(2020, 5, 1, tzinfo=pytz.utc)
    #     freezer.move_to(first_day_of_the_month)

    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!new"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     snap(channel.all_sent_responses[4])
    #     assert len(channel.all_sent_calls) == 5

    # async def test_on_message_bug(self, client, channel, without_bugs_header, snap):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(author, channel, "!bugs"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     assert len(channel.all_sent_calls) == 3

    # async def test_on_message_art_fulllist(self, client, channel, snap):
    #     await client.on_message(MockMessage(someone(), channel, "!art"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 1

    # async def test_on_message_art_correctnames(self, client, channel, snap):
    #     await client.on_message(
    #         MockMessage(someone(), channel, "!art amazing painting, proper painting",)
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 1

    # async def test_on_message_art_invalidnames(self, client, channel, snap):
    #     await client.on_message(
    #         MockMessage(someone(), channel, "!art academic painting, asdf",)
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 1

    # async def test_paginate(self, client):
    #     def subject(text):
    #         return [page for page in client.paginate(text)]

    #     assert subject("") == [""]
    #     assert subject("four") == ["four"]

    #     with open(Path(DAT_ROOT) / "ipsum_2011.txt") as f:
    #         text = f.read()
    #         pages = subject(text)
    #         assert len(pages) == 2
    #         assert all(len(page) <= 2000 for page in pages)
    #         assert pages == [text[0:1937], text[1937:]]

    #     with open(Path(DAT_ROOT) / "aaa_2001.txt") as f:
    #         text = f.read()
    #         pages = subject(text)
    #         assert len(pages) == 2
    #         assert all(len(page) <= 2000 for page in pages)
    #         assert pages == [text[0:2000], text[2000:]]

    #     with open(Path(DAT_ROOT) / "quotes.txt") as f:
    #         text = f.read()
    #         pages = subject(text)
    #         assert len(pages) == 2
    #         assert all(len(page) <= 2000 for page in pages)
    #         assert pages == [text[0:2000], f"> {text[2000:]}"]

    # async def test_humanize_months(self):
    #     def subject(*args):
    #         row = dict(
    #             zip(
    #                 [
    #                     "jan",
    #                     "feb",
    #                     "mar",
    #                     "apr",
    #                     "may",
    #                     "jun",
    #                     "jul",
    #                     "aug",
    #                     "sep",
    #                     "oct",
    #                     "nov",
    #                     "dec",
    #                 ],
    #                 args,
    #             )
    #         )
    #         return list(turbot.humanize_months(row))

    #     assert subject(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) == ["the entire year"]
    #     assert subject(1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0) == ["Jan - Mar", "Jul - Sep"]
    #     assert subject(1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1) == ["Jan - Mar", "Dec"]
    #     assert subject(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0) == ["Mar"]
    #     assert subject(0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1) == ["Mar", "Oct - Dec"]
    #     assert subject(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == ["Jan"]
    #     assert subject(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) == []

    # async def test_on_message_info(self, client, channel, snap):
    #     author = DUDE
    #     prefs = {
    #         "hemisphere": "norTHErn",
    #         "timezone": "America/Los_Angeles",
    #         "island": "Kriti",
    #         "friend": "Sw-1111----2222-3333",
    #         "fruit": "pEaCh",
    #         "nickname": "Phèdre nó Delaunay de Montrève",
    #         "creator": "ma---  4444----555 5-6666--",
    #     }
    #     for pref, value in prefs.items():
    #         await client.on_message(MockMessage(author, channel, f"!pref {pref} {value}"))
    #         await client.on_message(MockMessage(author, channel, f"!info {author.name}"))
    #     snap(channel.all_sent_embeds_json)
    #     assert len(channel.all_sent_calls) == len(prefs) * 2

    # async def test_get_user_prefs(self, client, channel):
    #     author = DUDE
    #     prefs = {
    #         "hemisphere": "norTHErn",
    #         "timezone": "America/Los_Angeles",
    #         "island": "Kriti",
    #         "friend": "Sw-1111----2222-3333",
    #         # "fruit": Not Set,
    #         "nickname": "Phèdre nó Delaunay de Montrève",
    #         "creator": "ma---  4444----555 5-6666--",
    #     }
    #     for pref, value in prefs.items():
    #         await client.on_message(MockMessage(author, channel, f"!pref {pref} {value}"))

    #     assert client.get_user_prefs(author.id) == {
    #         "friend": "111122223333",
    #         "hemisphere": "northern",
    #         "island": "Kriti",
    #         "nickname": "Phèdre nó Delaunay de Montrève",
    #         "timezone": pytz.timezone(prefs["timezone"]),
    #         "creator": "444455556666",
    #     }

    #     # unload in-memory users data
    #     client._users_data = None

    #     assert client.get_user_prefs(author.id) == {
    #         "friend": "111122223333",
    #         "hemisphere": "northern",
    #         "island": "Kriti",
    #         "nickname": "Phèdre nó Delaunay de Montrève",
    #         "timezone": pytz.timezone(prefs["timezone"]),
    #         "creator": "444455556666",
    #     }

    #     assert client.get_user_prefs(PUNK.id) == {}

    # async def test_get_user_prefs_friend_code(self, client, channel):
    #     author = someone()
    #     with open(client.data.file("users"), "w") as f:
    #         f.writelines(
    #             [
    #                 "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #                 f"{author.id},,,,111122223333.0,,,\n",
    #             ]
    #         )
    #     assert client.get_user_prefs(author.id) == {"friend": "111122223333.0"}
    #     await client.on_message(MockMessage(author, channel, f"!info {author.name}"))
    #     assert (
    #         next(
    #             f["value"]
    #             for f in channel.last_sent_embed["fields"]
    #             if f["name"] == "Friend code"
    #         )
    #         == "SW-1111-2222-3333"
    #     )

    # async def test_on_message_about(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!about"))
    #     assert len(channel.all_sent_calls) == 1

    #     about = channel.last_sent_embed
    #     assert about["title"] == "Turbot"
    #     assert about["url"] == "https://github.com/theastropath/turbot"
    #     assert about["description"] == (
    #         "A Discord bot for everything _Animal Crossing: New Horizons._\n\n"
    #         "Use the command `!help` for usage details. Having issues with Turbot? "
    #         "Please [report bugs](https://github.com/theastropath/turbot/issues)!\n"
    #     )
    #     assert about["footer"]["text"] == "MIT \u00a9 TheAstropath, lexicalunit et al"
    #     assert about["thumbnail"]["url"] == (
    #         "https://raw.githubusercontent.com/theastropath/turbot/master/turbot.png"
    #     )

    #     fields = {f["name"]: f["value"] for f in about["fields"]}
    #     assert fields["Package"] == "[PyPI](https://pypi.org/project/turbot/)"
    #     assert fields["Author"] == "[TheAstropath](https://github.com/theastropath)"
    #     assert fields["Maintainer"] == "[lexicalunit](https://github.com/lexicaluit)"

    #     version = turbot.__version__
    #     assert fields["Version"] == (
    #         f"[{version}](https://pypi.org/project/turbot/{version}/)"
    #     )

    # async def test_on_message_info_old_user(self, client, channel, monkeypatch):
    #     # Simulate the condition where a user exists in the data file,
    #     # but is no longer on the server.
    #     monkeypatch.setattr(turbot, "discord_user_name", lambda *_: None)

    #     await client.on_message(MockMessage(DUDE, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
    #     assert channel.last_sent_response == "No users found."

    # async def test_on_message_info_not_found(self, client, channel):
    #     await client.on_message(MockMessage(DUDE, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
    #     assert channel.last_sent_response == "No users found."

    # async def test_on_message_info_no_users(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
    #     assert channel.last_sent_response == "No users found."

    # async def test_on_message_info_no_prefs(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!buy 100"))
    #     await client.on_message(MockMessage(someone(), channel, f"!info {author.name}"))
    #     assert channel.last_sent_response == f"> **{author}** has no preferences."

    # async def test_on_message_info_no_params(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!info"))
    #     assert channel.last_sent_response == "Please provide a search term."

    # async def test_discord_user_from_name_guard(self, channel):
    #     assert turbot.discord_user_from_name(channel, None) == None

    # async def test_discord_user_name_guard(self, channel):
    #     assert turbot.discord_user_name(channel, None) == None

    # async def test_on_message_pref_no_params(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref"))
    #     assert channel.last_sent_response == (
    #         "Please provide a preference and a value, possible preferences include "
    #         "hemisphere, timezone, island, friend, fruit, nickname, creator."
    #     )

    # async def test_on_message_pref_no_value(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref creator"))
    #     assert channel.last_sent_response == (
    #         "Please provide the value for your creator preference."
    #     )

    # async def test_on_message_pref_invalid_pref(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref shazbot"))
    #     assert channel.last_sent_response == (
    #         "Please provide a valid preference name, possible preferences include "
    #         "hemisphere, timezone, island, friend, fruit, nickname, creator."
    #     )

    # async def test_on_message_pref_hemisphere_invalid(self, client, channel):
    #     await client.on_message(
    #         MockMessage(someone(), channel, "!pref hemisphere upwards")
    #     )
    #     assert channel.last_sent_response == (
    #         'Please provide either "northern" or "southern" as your hemisphere name.'
    #     )

    # async def test_on_message_pref_hemisphere(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere souTherN"))
    #     assert channel.last_sent_response == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},southern,,,,,,\n",
    #         ]

    #     await client.on_message(MockMessage(author, channel, "!pref hemisphere NoRthErn"))
    #     assert channel.last_sent_response == (
    #         f"Registered hemisphere preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},northern,,,,,,\n",
    #         ]

    # async def test_on_message_pref_friend_invalid(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref friend upwards"))
    #     assert channel.last_sent_response == (
    #         "Your switch friend code should be 12 numbers."
    #     )

    # async def test_on_message_pref_friend(self, client, channel):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(author, channel, "!pref friend sw-1234-5678-9012")
    #     )
    #     assert channel.last_sent_response == f"Registered friend preference for {author}."
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,,,123456789012,,,\n",
    #         ]

    # async def test_on_message_pref_creator_invalid(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref creator upwards"))
    #     assert channel.last_sent_response == (
    #         "Your Animal Crossing creator code should be 12 numbers."
    #     )

    # async def test_on_message_pref_creator(self, client, channel):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(author, channel, "!pref creator mA-1234-5678-9012")
    #     )
    #     assert (
    #         channel.last_sent_response == f"Registered creator preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,,,,,,123456789012\n",
    #         ]

    # async def test_on_message_pref_fruit_invalid(self, client, channel):
    #     await client.on_message(MockMessage(someone(), channel, "!pref fruit upwards"))
    #     assert channel.last_sent_response == (
    #         "Your native fruit can be apple, cherry, orange, peach, or pear."
    #     )

    # async def test_on_message_pref_fruit(self, client, channel):
    #     author = someone()
    #     await client.on_message(MockMessage(author, channel, "!pref fruit apple"))
    #     assert channel.last_sent_response == (
    #         f"Registered fruit preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,,,,apple,,\n",
    #         ]

    # async def test_on_message_pref_timezone_invalid(self, client, channel):
    #     await client.on_message(
    #         MockMessage(someone(), channel, "!pref timezone Mars/Noctis_City")
    #     )
    #     assert channel.last_sent_response == (
    #         "Please provide a valid timezone name, see "
    #         "https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for the "
    #         "complete list of TZ names."
    #     )

    # async def test_on_message_pref_timezone(self, client, channel):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(author, channel, "!pref timezone America/Los_Angeles")
    #     )
    #     assert channel.last_sent_response == (
    #         f"Registered timezone preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,America/Los_Angeles,,,,,\n",
    #         ]

    #     await client.on_message(
    #         MockMessage(author, channel, "!pref timezone Canada/Saskatchewan")
    #     )
    #     assert channel.last_sent_response == (
    #         f"Registered timezone preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,Canada/Saskatchewan,,,,,\n",
    #         ]

    # async def test_on_message_pref_island(self, client, channel):
    #     author = someone()
    #     island = "Koholint Island"
    #     await client.on_message(MockMessage(author, channel, f"!pref island {island}"))
    #     assert channel.last_sent_response == (
    #         f"Registered island preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,,{island},,,,\n",
    #         ]

    # async def test_on_message_pref_nickname(self, client, channel):
    #     author = someone()
    #     name = "Chuck Noland"
    #     await client.on_message(MockMessage(author, channel, f"!pref nickname {name}"))
    #     assert channel.last_sent_response == (
    #         f"Registered nickname preference for {author}."
    #     )
    #     with open(client.data.file("users")) as f:
    #         assert f.readlines() == [
    #             "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
    #             f"{author.id},,,,,,{name},\n",
    #         ]

    # async def test_on_message_uncollect_fish(self, client, channel, lines):
    #     # first collect some fossils
    #     author = someone()
    #     fish = "giant snakehead, snapping turtle ,bluegill"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))

    #     # then delete some of them
    #     fish = "giant snakehead, anime waifu, ancient statue, bluegill"
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {fish}"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following fish as collected:\n"
    #         "> bluegill, giant snakehead\n"
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> ancient statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("fish")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},snapping turtle\n"]

    #     # then delete the same ones again
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {fish}"))
    #     assert channel.last_sent_response == (
    #         "The following fish were already marked as not collected:\n"
    #         "> bluegill, giant snakehead\n"
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> ancient statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("fish")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},snapping turtle\n"]

    #     # and delete one more
    #     await client.on_message(
    #         MockMessage(author, channel, "!uncollect snapping turtle")
    #     )
    #     assert channel.last_sent_response == (
    #         "Unmarked the following fish as collected:\n" "> snapping turtle"
    #     )
    #     with open(client.data.file("fish")) as f:
    #         assert f.readlines() == ["author,name\n"]

    # async def test_on_message_search_fish_no_need_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect snapping turtle, giant snakehead")
    #     )
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, "!collect snapping turtle, giant snakehead")
    #     )
    #     await client.on_message(
    #         MockMessage(
    #             GUY,
    #             channel,
    #             "!collect snapping turtle, giant snakehead, wistful painting",
    #         )
    #     )

    #     await client.on_message(
    #         MockMessage(
    #             PUNK, channel, "!search snapping turtle, giant snakehead, anime waifu"
    #         )
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: giant snakehead, snapping turtle\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_search_fish(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect snapping turtle, giant snakehead")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect snapping turtle"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect snapping turtle, giant snakehead")
    #     )

    #     query = "snapping turtle, giant snakehead, wistful painting"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     channel.last_sent_response == (
    #         "__**Fish Search**__\n"
    #         f"> {BUDDY} needs: giant snakehead, wistful painting\n"
    #         f"> {FRIEND} needs: wistful painting\n"
    #         f"> {GUY} needs: wistful painting"
    #     )

    # async def test_on_message_search_fish_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect snapping turtle, giant snakehead")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect snapping turtle"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect snapping turtle, giant snakehead")
    #     )

    #     query = "snapping turtle, giant snakehead, wistful painting, anime waifu"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     assert channel.last_sent_response == (
    #         "> No one needs: snapping turtle, wistful painting\n"
    #         f"> {BUDDY} needs fish: giant snakehead\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_count_fish(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect snapping turtle, giant snakehead")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect snapping turtle"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect snapping turtle, giant snakehead")
    #     )

    #     users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
    #     await client.on_message(MockMessage(author, channel, f"!count {users}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_fish_no_name(self, client, channel):
    #     author = DUDE
    #     fish = "snapping turtle, bluegill, giant snakehead"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 fish donated by {DUDE}**__\n"
    #         "> bluegill, giant snakehead, snapping turtle"
    #     )

    # async def test_on_message_collected_fish_congrats(self, client, channel):
    #     everything = ",".join(client.assets["fish"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!collected"))
    #     channel.last_sent_response == "**Congratulations, you've collected all fish!**"

    # async def test_on_message_uncollected_fish_congrats(self, client, channel, snap):
    #     everything = ",".join(client.assets["fish"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!uncollected"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_fish_with_name(self, client, channel):
    #     fish = "snapping turtle, bluegill, giant snakehead"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fish}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**3 fish donated by {GUY}**__\n"
    #         "> bluegill, giant snakehead, snapping turtle"
    #     )

    # async def test_on_message_collected_fish_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_collect_fish(self, client, channel, lines):
    #     # first collect some fish
    #     author = BUDDY
    #     fish = "bluegill, snapping turtle, anime waifu"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following fish as collected:\n"
    #         "> bluegill, snapping turtle\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     assert set(lines(client.data.file("fish"))) == {
    #         "author,name\n",
    #         f"{author.id},bluegill\n",
    #         f"{author.id},snapping turtle\n",
    #     }

    #     # collect them again
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))
    #     assert channel.last_sent_response == (
    #         "The following fish had already been collected:\n"
    #         "> bluegill, snapping turtle\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )

    #     # collect some new stuff, but with some dupes
    #     fish = "body pillow, snapping turtle, tadpole"
    #     await client.on_message(MockMessage(author, channel, f"!collect {fish}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following fish as collected:\n"
    #         "> tadpole\n"
    #         "The following fish had already been collected:\n"
    #         "> snapping turtle\n"
    #         "Unrecognized collectable names:\n"
    #         "> body pillow"
    #     )

    #     assert lines(client.data.file("fish")) == [f"{author.id},tadpole\n"]

    # async def test_on_message_collect_fish_congrats(self, client, channel, snap):
    #     everything = sorted(list(client.assets["fish"].all))
    #     some, rest = everything[:10], everything[10:]

    #     # someone else collects some fish
    #     fish = "bluegill, snapping turtle, tadpole"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {fish}"))

    #     # Buddy collects some
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
    #     )

    #     # Friend collects a different set
    #     fish = "giant snakehead, black bass"
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {fish}"))

    #     # Buddy collects the rest
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(rest)}")
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_fish_none_available(self, client, channel):
    #     everything = ",".join(client.assets["fish"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!fish"))
    #     assert channel.last_sent_response == (
    #         "No fish that you haven't already caught are available at this time."
    #     )
    #     assert len(channel.all_sent_calls) == 3

    # async def test_creatures_available_now(self, client):
    #     def creature(name, time):
    #         return ["northern", name, "image-url", 50, "everywhere", time] + [1] * 12

    #     import pandas as pd

    #     creatures = pd.DataFrame(
    #         columns=client.assets["bugs"].data.columns,
    #         data=[
    #             creature("one", "1 am - 10 am"),
    #             creature("two", "1 am - 10 pm"),
    #             creature("three", "1 pm - 10 am"),
    #             creature("four", "1 pm - 10 pm"),
    #             creature("five", "10 am - 1 am"),
    #             creature("six", "10 am - 1 pm"),
    #             creature("seven", "10 pm - 1 am"),
    #             creature("eight", "10 pm - 1 pm"),
    #             creature("nine", "16 pm - 10 am"),
    #             creature("nine", "13 pm - 17 pm"),
    #             creature("nine", "all day"),
    #             creature("ten", "1 am - 3 am & 6 am - 10 am"),
    #         ],
    #     )

    #     def subject(dt):
    #         return set(client.creatures_available_now(dt, creatures))

    #     assert subject(datetime(2020, 4, 6, 0)) == {"three", "seven", "nine"}
    #     assert subject(datetime(2020, 4, 6, 1)) == {
    #         "three",
    #         "ten",
    #         "seven",
    #         "two",
    #         "nine",
    #         "one",
    #     }
    #     assert subject(datetime(2020, 4, 6, 2)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 3)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 4)) == {"one", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 5)) == {"one", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 6)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 7)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 8)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 9)) == {"three", "ten", "two", "nine", "one"}
    #     assert subject(datetime(2020, 4, 6, 10)) == {
    #         "three",
    #         "ten",
    #         "one",
    #         "two",
    #         "nine",
    #         "six",
    #     }
    #     assert subject(datetime(2020, 4, 6, 11)) == {"six", "nine", "two"}
    #     assert subject(datetime(2020, 4, 6, 12)) == {
    #         "three",
    #         "seven",
    #         "two",
    #         "nine",
    #         "six",
    #     }
    #     assert subject(datetime(2020, 4, 6, 13)) == {
    #         "three",
    #         "seven",
    #         "two",
    #         "nine",
    #         "six",
    #     }
    #     assert subject(datetime(2020, 4, 6, 14)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 15)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 16)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 17)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 18)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 19)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 20)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 21)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 22)) == {"seven", "nine", "three", "two"}
    #     assert subject(datetime(2020, 4, 6, 23)) == {"seven", "nine", "three"}

    # async def test_on_message_uncollect_bugs(self, client, channel, lines):
    #     # first collect some bugs
    #     author = someone()
    #     bugs = "great purple emperor, stinkbug ,bell cricket"
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))

    #     # then delete some of them
    #     bugs = "great purple emperor, anime waifu, ancient statue, bell cricket"
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {bugs}"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following bugs as collected:\n"
    #         "> bell cricket, great purple emperor\n"
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> ancient statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("bugs")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},stinkbug\n"]

    #     # then delete the same ones again
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {bugs}"))
    #     assert channel.last_sent_response == (
    #         "The following bugs were already marked as not collected:\n"
    #         "> bell cricket, great purple emperor\n"
    #         "The following pieces of art were already marked as not collected:\n"
    #         "> ancient statue\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("bugs")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},stinkbug\n"]

    #     # and delete one more
    #     await client.on_message(MockMessage(author, channel, "!uncollect stinkbug"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following bugs as collected:\n" "> stinkbug"
    #     )
    #     with open(client.data.file("bugs")) as f:
    #         assert f.readlines() == ["author,name\n"]

    # async def test_on_message_search_bugs_no_need_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect stinkbug, great purple emperor")
    #     )
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, "!collect stinkbug, great purple emperor")
    #     )
    #     await client.on_message(
    #         MockMessage(
    #             GUY, channel, "!collect stinkbug, great purple emperor, wistful painting",
    #         )
    #     )

    #     await client.on_message(
    #         MockMessage(
    #             PUNK, channel, "!search stinkbug, great purple emperor, anime waifu"
    #         )
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: great purple emperor, stinkbug\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_search_bugs(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect stinkbug, great purple emperor")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect stinkbug"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect stinkbug, great purple emperor")
    #     )

    #     query = "stinkbug, great purple emperor, wistful painting"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     channel.last_sent_response == (
    #         "__**Bugs Search**__\n"
    #         f"> {BUDDY} needs: great purple emperor, wistful painting\n"
    #         f"> {FRIEND} needs: wistful painting\n"
    #         f"> {GUY} needs: wistful painting"
    #     )

    # async def test_on_message_search_bugs_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect stinkbug, great purple emperor")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect stinkbug"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect stinkbug, great purple emperor")
    #     )

    #     query = "stinkbug, great purple emperor, wistful painting, anime waifu"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     assert channel.last_sent_response == (
    #         "> No one needs: stinkbug, wistful painting\n"
    #         f"> {BUDDY} needs bugs: great purple emperor\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_count_bugs(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect stinkbug, great purple emperor")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect stinkbug"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect stinkbug, great purple emperor")
    #     )

    #     users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
    #     await client.on_message(MockMessage(author, channel, f"!count {users}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_bugs_no_name(self, client, channel):
    #     author = DUDE
    #     bugs = "stinkbug, bell cricket, great purple emperor"
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 bugs donated by {DUDE}**__\n"
    #         "> bell cricket, great purple emperor, stinkbug"
    #     )

    # async def test_on_message_collected_bugs_congrats(self, client, channel):
    #     everything = ",".join(client.assets["bugs"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!collected"))
    #     channel.last_sent_response == "**Congratulations, you've collected all bugs!**"

    # async def test_on_message_uncollected_bugs_congrats(self, client, channel, snap):
    #     everything = ",".join(client.assets["bugs"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!uncollected"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_bugs_with_name(self, client, channel):
    #     bugs = "stinkbug, bell cricket, great purple emperor"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {bugs}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**3 bugs donated by {GUY}**__\n"
    #         "> bell cricket, great purple emperor, stinkbug"
    #     )

    # async def test_on_message_collected_bugs_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_collect_bugs(self, client, channel, lines):
    #     # first collect some bugs
    #     author = BUDDY
    #     bugs = "bell cricket, stinkbug, anime waifu"
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following bugs as collected:\n"
    #         "> bell cricket, stinkbug\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     assert set(lines(client.data.file("bugs"))) == {
    #         "author,name\n",
    #         f"{author.id},bell cricket\n",
    #         f"{author.id},stinkbug\n",
    #     }

    #     # collect them again
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))
    #     assert channel.last_sent_response == (
    #         "The following bugs had already been collected:\n"
    #         "> bell cricket, stinkbug\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )

    #     # collect some new stuff, but with some dupes
    #     bugs = "body pillow, stinkbug, tiger beetle"
    #     await client.on_message(MockMessage(author, channel, f"!collect {bugs}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following bugs as collected:\n"
    #         "> tiger beetle\n"
    #         "The following bugs had already been collected:\n"
    #         "> stinkbug\n"
    #         "Unrecognized collectable names:\n"
    #         "> body pillow"
    #     )

    #     assert lines(client.data.file("bugs")) == [f"{author.id},tiger beetle\n"]

    # async def test_on_message_collect_bugs_congrats(self, client, channel, snap):
    #     everything = sorted(list(client.assets["bugs"].all))
    #     some, rest = everything[:10], everything[10:]

    #     # someone else collects some bugs
    #     bugs = "bell cricket, stinkbug, tiger beetle"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {bugs}"))

    #     # Buddy collects some
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
    #     )

    #     # Friend collects a different set
    #     bugs = "great purple emperor, banded dragonfly"
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {bugs}"))

    #     # Buddy collects the rest
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(rest)}")
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_bugs_none_available(self, client, channel):
    #     everything = ",".join(client.assets["bugs"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))
    #     await client.on_message(MockMessage(BUDDY, channel, "!bugs"))
    #     assert channel.last_sent_response == (
    #         "No bugs that you haven't already caught are available at this time."
    #     )
    #     assert len(channel.all_sent_calls) == 3

    # async def test_on_message_uncollect_songs(self, client, channel, lines):
    #     # first collect some songs
    #     author = someone()
    #     songs = "k.k. groove, k.k. safari ,k.k. bazaar"
    #     await client.on_message(MockMessage(author, channel, f"!collect {songs}"))

    #     # then delete some of them
    #     songs = "k.k. groove, anime waifu, k.k. aria, k.k. bazaar"
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {songs}"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following songs as collected:\n"
    #         "> K.K. Bazaar, K.K. Groove\n"
    #         "The following songs were already marked as not collected:\n"
    #         "> K.K. Aria\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("songs")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},K.K. Safari\n"]

    #     # then delete the same ones again
    #     await client.on_message(MockMessage(author, channel, f"!uncollect {songs}"))
    #     assert channel.last_sent_response == (
    #         "The following songs were already marked as not collected:\n"
    #         "> K.K. Aria, K.K. Bazaar, K.K. Groove\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     with open(client.data.file("songs")) as f:
    #         assert f.readlines() == ["author,name\n", f"{author.id},K.K. Safari\n"]

    #     # and delete one more
    #     await client.on_message(MockMessage(author, channel, "!uncollect k.k. safari"))
    #     assert channel.last_sent_response == (
    #         "Unmarked the following songs as collected:\n" "> K.K. Safari"
    #     )
    #     with open(client.data.file("songs")) as f:
    #         assert f.readlines() == ["author,name\n"]

    # async def test_on_message_search_songs_no_need_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect k.k. safari, k.k. groove")
    #     )
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, "!collect k.k. safari, k.k. groove")
    #     )
    #     await client.on_message(
    #         MockMessage(
    #             GUY, channel, "!collect k.k. safari, k.k. groove, wistful painting",
    #         )
    #     )

    #     await client.on_message(
    #         MockMessage(PUNK, channel, "!search k.k. safari, k.k. groove, anime waifu")
    #     )
    #     assert channel.last_sent_response == (
    #         "> No one needs: K.K. Groove, K.K. Safari\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_search_songs(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect k.k. safari, k.k. groove")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect k.k. safari"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect k.k. safari, k.k. groove")
    #     )

    #     query = "k.k. safari, k.k. groove, wistful painting"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     channel.last_sent_response == (
    #         "__**songs Search**__\n"
    #         f"> {BUDDY} needs: K.K. Groove, wistful painting\n"
    #         f"> {FRIEND} needs: wistful painting\n"
    #         f"> {GUY} needs: wistful painting"
    #     )

    # async def test_on_message_search_songs_with_bad(self, client, channel):
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect k.k. safari, k.k. groove")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect k.k. safari"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect k.k. safari, k.k. groove")
    #     )

    #     query = "k.k. safari, k.k. groove, wistful painting, anime waifu"
    #     await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
    #     assert channel.last_sent_response == (
    #         "> No one needs: K.K. Safari, wistful painting\n"
    #         f"> {BUDDY} needs songs: K.K. Groove\n"
    #         "Did not recognize the following collectables:\n"
    #         "> anime waifu"
    #     )

    # async def test_on_message_count_songs(self, client, channel, snap):
    #     author = someone()
    #     await client.on_message(
    #         MockMessage(FRIEND, channel, "!collect k.k. safari, k.k. groove")
    #     )
    #     await client.on_message(MockMessage(BUDDY, channel, "!collect k.k. safari"))
    #     await client.on_message(
    #         MockMessage(GUY, channel, "!collect k.k. safari, k.k. groove")
    #     )

    #     users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
    #     await client.on_message(MockMessage(author, channel, f"!count {users}"))
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_songs_no_name(self, client, channel):
    #     author = DUDE
    #     songs = "k.k. safari, k.k. bazaar, k.k. groove"
    #     await client.on_message(MockMessage(author, channel, f"!collect {songs}"))

    #     await client.on_message(MockMessage(author, channel, "!collected"))
    #     assert channel.last_sent_response == (
    #         f"__**3 songs collected by {DUDE}**__\n"
    #         "> K.K. Bazaar, K.K. Groove, K.K. Safari"
    #     )

    # async def test_on_message_collected_songs_congrats(self, client, channel):
    #     everything = ",".join(client.assets["songs"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!collected"))
    #     channel.last_sent_response == "**Congratulations, you've collected all songs!**"

    # async def test_on_message_uncollected_songs_congrats(self, client, channel, snap):
    #     everything = ",".join(client.assets["songs"].all)
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

    #     await client.on_message(MockMessage(BUDDY, channel, "!uncollected"))
    #     snap(channel.all_sent_responses[1])
    #     snap(channel.all_sent_responses[2])
    #     snap(channel.all_sent_responses[3])
    #     assert len(channel.all_sent_calls) == 4

    # async def test_on_message_collected_songs_with_name(self, client, channel):
    #     songs = "safari, bazaar, groove"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {songs}"))

    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
    #     assert channel.last_sent_response == (
    #         f"__**3 songs collected by {GUY}**__\n"
    #         "> K.K. Bazaar, K.K. Groove, K.K. Safari"
    #     )

    # async def test_on_message_collected_songs_bad_name(self, client, channel):
    #     await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
    #     assert channel.last_sent_response == (
    #         f"Can not find the user named {PUNK.name} in this channel."
    #     )

    # async def test_on_message_collect_songs(self, client, channel, lines):
    #     # first collect some songs
    #     author = BUDDY
    #     songs = "k.k. bazaar, k.k. safari, anime waifu"
    #     await client.on_message(MockMessage(author, channel, f"!collect {songs}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following songs as collected:\n"
    #         "> K.K. Bazaar, K.K. Safari\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )
    #     assert set(lines(client.data.file("songs"))) == {
    #         "author,name\n",
    #         f"{author.id},K.K. Bazaar\n",
    #         f"{author.id},K.K. Safari\n",
    #     }

    #     # collect them again
    #     await client.on_message(MockMessage(author, channel, f"!collect {songs}"))
    #     assert channel.last_sent_response == (
    #         "The following songs had already been collected:\n"
    #         "> K.K. Bazaar, K.K. Safari\n"
    #         "Unrecognized collectable names:\n"
    #         "> anime waifu"
    #     )

    #     # collect some new stuff, but with some dupes
    #     songs = "body pillow, k.k. safari, k.k. tango"
    #     await client.on_message(MockMessage(author, channel, f"!collect {songs}"))
    #     assert channel.last_sent_response == (
    #         "Marked the following songs as collected:\n"
    #         "> K.K. Tango\n"
    #         "The following songs had already been collected:\n"
    #         "> K.K. Safari\n"
    #         "Unrecognized collectable names:\n"
    #         "> body pillow"
    #     )

    #     assert lines(client.data.file("songs")) == [f"{author.id},K.K. Tango\n"]

    # async def test_on_message_collect_songs_congrats(self, client, channel, snap):
    #     everything = sorted(list(client.assets["songs"].all))
    #     some, rest = everything[:10], everything[10:]

    #     # someone else collects some songs
    #     songs = "k.k. bazaar, k.k. safari, k.k. tango"
    #     await client.on_message(MockMessage(GUY, channel, f"!collect {songs}"))

    #     # Buddy collects some
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
    #     )

    #     # Friend collects a different set
    #     songs = "k.k. groove, comrade k.k."
    #     await client.on_message(MockMessage(FRIEND, channel, f"!collect {songs}"))

    #     # Buddy collects the rest
    #     await client.on_message(
    #         MockMessage(BUDDY, channel, f"!collect {', '.join(rest)}")
    #     )
    #     snap(channel.last_sent_response)
    #     assert len(channel.all_sent_calls) == 4

    async def test_client_data_exception(self, client):
        with pytest.raises(RuntimeError):
            client.data.foobar


class TestFigures:
    # Some realistic price data sampled from the wild:
    PRICES = [
        [FRIEND.id, "buy", 103, "2020-04-05 09:00:00+00:00"],  # Sunday_AM
        [FRIEND.id, "sell", 112, "2020-04-06 09:00:00+00:00"],  # Monday_AM
        [FRIEND.id, "sell", 116, "2020-04-06 13:00:00+00:00"],  # Monday_PM
        [FRIEND.id, "sell", 80, "2020-04-07 09:00:00+00:00"],  # Tuesday_AM
        # [FRIEND.id, 'sell', None, "2020-04-07 13:00:00+00:00"],  # Tuesday_PM
        [FRIEND.id, "sell", 100, "2020-04-08 09:00:00+00:00"],  # Wednesday_AM
        # [FRIEND.id, 'sell', None, "2020-04-08 13:00:00+00:00"],  # Wednesday_PM
        [FRIEND.id, "sell", 95, "2020-04-09 09:00:00+00:00"],  # Thursday_AM
        # [FRIEND.id, 'sell', None, "2020-04-09 13:00:00+00:00"],  # Thursday_PM
        [FRIEND.id, "sell", 80, "2020-04-10 09:00:00+00:00"],  # Friday_AM
        # [FRIEND.id, 'sell', None, "2020-04-10 13:00:00+00:00"],  # Friday_PM
        # [FRIEND.id, 'sell', None, "2020-04-11 09:00:00+00:00"],  # Saturday_AM
        # [FRIEND.id, 'sell', None, "2020-04-11 13:00:00+00:00"],  # Saturday_PM
        [DUDE.id, "buy", 98, "2020-04-05 09:00:00+00:00"],  # Sunday_AM
        [DUDE.id, "sell", 88, "2020-04-06 09:00:00+00:00"],  # Monday_AM
        [DUDE.id, "sell", 84, "2020-04-06 13:00:00+00:00"],  # Monday_PM
        [DUDE.id, "sell", 81, "2020-04-07 09:00:00+00:00"],  # Tuesday_AM
        [DUDE.id, "sell", 76, "2020-04-07 13:00:00+00:00"],  # Tuesday_PM
        # [DUDE.id, 'sell', None, "2020-04-08 09:00:00+00:00"],  # Wednesday_AM
        # [DUDE.id, 'sell', None, "2020-04-08 13:00:00+00:00"],  # Wednesday_PM
        [DUDE.id, "sell", 138, "2020-04-09 09:00:00+00:00"],  # Thursday_AM
        [DUDE.id, "sell", 336, "2020-04-09 13:00:00+00:00"],  # Thursday_PM
        [DUDE.id, "sell", 191, "2020-04-10 09:00:00+00:00"],  # Friday_AM
        [DUDE.id, "sell", 108, "2020-04-10 13:00:00+00:00"],  # Friday_PM
        # [DUDE.id, 'sell', None, "2020-04-11 09:00:00+00:00"],  # Saturday_AM
        # [DUDE.id, 'sell', None, "2020-04-11 13:00:00+00:00"],  # Saturday_PM
    ]

    def set_example_prices(self, client):
        c = client.data.conn.cursor()
        values = [f"({row[0]}, '{row[1]}', {row[2]}, '{row[3]}')" for row in self.PRICES]
        c.execute(
            f"""
            INSERT INTO prices (author, kind, price, timestamp)
            VALUES
            {','.join(values)};
            """
        )

    def set_bogus_prices(self, client):
        c = client.data.conn.cursor()
        c.execute(
            f"""
            INSERT INTO prices (author, kind, price, timestamp)
            VALUES
            /* user that's not in the channel: */
            ({PUNK.id}, 'buy', 100, '1982-04-24 01:00:00+00:00'),

            /* some actually valid data: */
            ({FRIEND.id}, 'buy', 103,'2020-04-05 09:00:00+00:00'),
            ({FRIEND.id}, 'sell', 112,'2020-04-06 09:00:00+00:00'),

            /* a user with only buy data, no sell data: */
            ({DUDE.id}, 'buy', 98,'2020-04-05 09:00:00+00:00');
            """
        )

    def set_error_prices(self, client):
        c = client.data.conn.cursor()
        c.execute(
            f"""
            INSERT INTO prices (author, kind, price, timestamp)
            VALUES
            /* for some reason this sequence of prices errors out the turnips lib: */
            ({BUDDY.id}, 'buy', 102, '2020-04-05 09:00:00+00:00'),
            ({BUDDY.id}, 'sell', 93, '2020-04-06 09:00:00+00:00'),
            ({BUDDY.id}, 'sell', 87, '2020-04-06 13:00:00+00:00'),
            ({BUDDY.id}, 'sell', 86, '2020-04-07 09:00:00+00:00'),
            ({BUDDY.id}, 'sell', 79, '2020-04-07 13:00:00+00:00'),
            ({BUDDY.id}, 'sell', 69, '2020-04-08 13:00:00+00:00');
            """
        )

    def test_get_graph_predictive_error(self, client, channel):
        self.set_error_prices(client)
        client.get_graph(channel, BUDDY, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    def test_get_graph_predictive_bad_user(self, client, channel):
        self.set_example_prices(client)
        client.get_graph(channel, PUNK, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    def test_get_graph_historical_no_users(self, client, channel):
        client.get_graph(channel, None, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    def test_get_graph_predictive_no_data(self, client, channel):
        client.get_graph(channel, FRIEND, turbot.GRAPHCMD_FILE)
        assert not Path(turbot.GRAPHCMD_FILE).exists()

    @pytest.mark.mpl_image_compare
    def test_get_graph_historical_with_bogus_data(self, client, channel):
        self.set_bogus_prices(client)
        client.get_graph(channel, None, turbot.GRAPHCMD_FILE)
        return client.get_graph(channel, None, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_historical(self, client, channel):
        self.set_example_prices(client)
        return client.get_graph(channel, None, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_predictive_friend(self, client, channel):
        self.set_example_prices(client)
        return client.get_graph(channel, FRIEND, turbot.GRAPHCMD_FILE)

    @pytest.mark.mpl_image_compare
    def test_get_graph_predictive_dude(self, client, channel):
        self.set_example_prices(client)
        return client.get_graph(channel, DUDE, turbot.GRAPHCMD_FILE)


class TestCodebase:
    def test_flake8(self):
        """Checks that the Python codebase passes configured flake8 checks."""
        chdir(REPO_ROOT)
        cmd = ["flake8", *SRC_DIRS]
        print("running:", " ".join(str(part) for part in cmd))
        proc = run(cmd, capture_output=True)
        assert proc.returncode == 0, f"flake8 issues:\n{proc.stdout.decode('utf-8')}"

    def test_black(self):
        """Checks that the Python codebase passes configured black checks."""
        chdir(REPO_ROOT)
        cmd = ["black", "-v", "--check", *SRC_DIRS]
        print("running:", " ".join(str(part) for part in cmd))
        proc = run(cmd, capture_output=True)
        assert proc.returncode == 0, f"black issues:\n{proc.stderr.decode('utf-8')}"

    def test_isort(self):
        """Checks that the Python codebase imports are correctly sorted."""
        chdir(REPO_ROOT)
        cmd = ["isort", "-df", "-rc", "-c", *SRC_DIRS]
        print("running:", " ".join(str(part) for part in cmd))
        proc = run(cmd, capture_output=True)
        assert proc.returncode == 0, f"isort issues:\n{proc.stdout.decode('utf-8')}"

    def test_sort_strings(self):
        """Checks that the strings data is correctly sorted."""
        chdir(REPO_ROOT)
        cmd = ["python", "scripts/sort_strings.py", "--check"]
        print("running:", " ".join(str(part) for part in cmd))
        proc = run(cmd, capture_output=True)
        assert proc.returncode == 0, (
            f"sort strings issues:\n{proc.stdout.decode('utf-8')}\n\n"
            "Please run ./scripts/sort_string.py to resolve this issue."
        )

    def test_snapshots_size(self):
        """Checks that none of the snapshots files are unreasonably small."""
        snapshots_dir = REPO_ROOT / "tests" / "snapshots"
        small_snapshots = []
        for f in snapshots_dir.glob("*.txt"):
            if f.stat().st_size <= 150:
                small_snapshots.append(f"- {f.name}")
        if small_snapshots:
            offenders = "\n".join(small_snapshots)
            assert False, (
                "Very small snapshot files are problematic.\n"
                "Offending snapshot files:\n"
                f"{offenders}\n"
                "Consider refacotring them to avoid using snapshots. Tests that use "
                "snapshots are harder to reason about when they fail. Whenever possilbe "
                "a test with inline data is much easier to reason about and refactor."
            )

    def test_readme_commands(self, client):
        """Checks that all commands are documented in our readme."""
        with open(REPO_ROOT / "README.md") as f:
            readme = f.read()

        documented = set(re.findall("^- `!([a-z]+)`: .*$", readme, re.MULTILINE))
        implemented = set(client.commands)

        assert documented == implemented

    def test_pyproject_dependencies(self):
        """Checks that pyproject.toml dependencies are sorted."""
        pyproject = toml.load("pyproject.toml")

        dev_deps = list(pyproject["tool"]["poetry"]["dev-dependencies"].keys())
        assert dev_deps == sorted(dev_deps)

        deps = list(pyproject["tool"]["poetry"]["dependencies"].keys())
        assert deps == sorted(deps)


# These tests will fail in isolation, you must run the full test suite for them to pass.
class TestMeta:
    # Tracks the usage of string keys over the entire test session.
    # It can fail for two reasons:
    #
    # 1. There's a key in strings.yaml that's not being used at all.
    # 2. There's a key in strings.yaml that isn't being used in the tests.
    #
    # For situation #1 the solution is to remove the key from the config.
    # As for #2, there should be a new test which utilizes this key.
    def test_strings(self):
        """Assues that there are no missing or unused strings data."""
        used_keys = set(s_call[0][0] for s_call in S_SPY.call_args_list)
        config_keys = set(load_strings().keys())
        assert config_keys - used_keys == set()

    # Tracks the usage of snapshot files over the entire test session.
    # When it fails it means you probably need to clear out any unused snapshot files.
    def test_snapshots(self):
        """Checks that all of the snapshots files are being used."""
        snapshots_dir = REPO_ROOT / "tests" / "snapshots"
        snapshot_files = set(f.name for f in snapshots_dir.glob("*.txt"))
        assert snapshot_files == SNAPSHOTS_USED
