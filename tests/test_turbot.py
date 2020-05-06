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
from callee import Matching

import turbot

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

    async def test_on_message_sell_at_time_with_tz(self, client, channel, lines, freezer):
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
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},sell,{amount},{monday_evening_adjust}\n",
        ]

    async def test_get_user_timeline(self, client, channel, lines, freezer):
        author = someone()
        author_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
        )

        sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
        freezer.move_to(sunday_am)
        await client.on_message(MockMessage(author, channel, f"!buy 90"))

        amount = 100
        monday_am = sunday_am + timedelta(days=1)
        for offset in range(0, 50):
            freezer.move_to(monday_am + timedelta(hours=offset))
            await client.on_message(
                MockMessage(author, channel, f"!sell {amount} monday morning")
            )
            amount += 5

        assert client.get_user_timeline(author.id) == [
            90,
            345,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    async def test_get_user_timeline_buy_monday(self, client, channel, lines, freezer):
        author = someone()
        author_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
        )

        sunday_am = datetime(2020, 4, 27, 9, tzinfo=pytz.utc)
        freezer.move_to(sunday_am)
        await client.on_message(MockMessage(author, channel, f"!buy 90"))

        amount = 100
        monday_am = sunday_am + timedelta(days=1)
        for offset in range(0, 50):
            freezer.move_to(monday_am + timedelta(hours=offset))
            await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
            amount += 5

        assert client.get_user_timeline(author.id) == [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    async def test_get_user_timeline_no_sells(self, client, channel, lines, freezer):
        author = someone()
        author_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
        )

        sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
        freezer.move_to(sunday_am)
        await client.on_message(MockMessage(author, channel, f"!buy 90"))

        assert client.get_user_timeline(author.id) == [
            90,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    async def test_get_user_timeline_sunday_sells(self, client, channel, lines, freezer):
        author = someone()
        author_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {author_tz.zone}")
        )

        sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
        freezer.move_to(sunday_am)
        await client.on_message(MockMessage(author, channel, f"!buy 90"))

        amount = 100
        monday_am = sunday_am + timedelta(days=1)
        for offset in range(0, 50):
            freezer.move_to(monday_am + timedelta(hours=offset))
            await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
            amount += 5

        await client.on_message(MockMessage(author, channel, f"!sell 50 sunday evening"))

        assert client.get_user_timeline(author.id) == [
            90,
            145,
            205,
            265,
            325,
            345,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    async def test_on_message_sell_at_time(self, client, channel, lines, freezer):
        monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
        monday_evening = monday_morning + timedelta(hours=13)
        command_time = monday_morning + timedelta(days=3)
        freezer.move_to(command_time)

        author = someone()
        amount = somebells()
        await client.on_message(
            MockMessage(author, channel, f"!sell {amount} monday evening")
        )
        assert channel.last_sent_response == (
            f"Logged selling price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},sell,{amount},{monday_evening}\n",
        ]

    async def test_on_message_sell_bad_time(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell 100 funday"))
        assert channel.last_sent_response == (
            "Please provide both the day of the week and time of day."
        )

    async def test_on_message_sell_bad_day(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell 100 fun morning"))
        assert channel.last_sent_response == (
            "Please use monday, wednesday, tuesday, etc for the day parameter."
        )

    async def test_on_message_sell_incomplete_time(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell 100 friday pants"))
        assert channel.last_sent_response == (
            "Please use either morning or evening as the time parameter."
        )

    async def test_on_message_sell_no_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell"))
        assert channel.last_sent_response == (
            "Please include selling price after command name."
        )

    async def test_on_message_sell_nonnumeric_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell foot"))
        assert channel.last_sent_response == ("Selling price must be a number.")

    async def test_on_message_sell_nonpositive_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!sell 0"))
        assert channel.last_sent_response == ("Selling price must be greater than zero.")

    async def test_on_message_sell_extra_space(self, client, channel, lines):
        author = someone()
        amount = somebells()
        await client.on_message(MockMessage(author, channel, f"!sell  {amount}"))
        assert channel.last_sent_response == (
            f"Logged selling price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},sell,{amount},{NOW}\n",
        ]

    async def test_on_message_sell(self, client, channel, lines):
        # initial sale
        author = someone()
        amount = somebells()
        await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
        assert channel.last_sent_response == (
            f"Logged selling price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},sell,{amount},{NOW}\n",
        ]

        # same price sale
        await client.on_message(MockMessage(author, channel, f"!sell {amount}"))
        assert channel.last_sent_response == (
            f"Logged selling price of {amount} for user {author}. "
            f"(Same as last selling price)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{amount},{NOW}\n"]

        # higher price sale
        new_amount = amount + somebells()
        await client.on_message(MockMessage(author, channel, f"!sell {new_amount}"))
        assert channel.last_sent_response == (
            f"Logged selling price of {new_amount} for user {author}. "
            f"(Higher than last selling price of {amount} bells)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{new_amount},{NOW}\n"]

        # lower price sale
        last_amount = round(amount / 2)
        await client.on_message(MockMessage(author, channel, f"!sell {last_amount}"))
        assert channel.last_sent_response == (
            f"Logged selling price of {last_amount} for user {author}. "
            f"(Lower than last selling price of {new_amount} bells)"
        )
        assert lines(client.prices_file) == [f"{author.id},sell,{last_amount},{NOW}\n"]

    async def test_on_message_buy_at_time_with_tz(self, client, channel, lines, freezer):
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
            MockMessage(author, channel, f"!buy {amount} monday evening")
        )
        assert channel.last_sent_response == (
            f"Logged buying price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},buy,{amount},{monday_evening_adjust}\n",
        ]

    async def test_on_message_buy_at_time(self, client, channel, lines, freezer):
        monday_morning = datetime(1982, 4, 19, tzinfo=pytz.utc)
        monday_evening = monday_morning + timedelta(hours=13)
        command_time = monday_morning + timedelta(days=3)
        freezer.move_to(command_time)

        author = someone()
        amount = somebells()
        await client.on_message(
            MockMessage(author, channel, f"!buy {amount} monday evening")
        )
        assert channel.last_sent_response == (
            f"Logged buying price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},buy,{amount},{monday_evening}\n",
        ]

    async def test_on_message_buy_bad_time(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy 100 funday"))
        assert channel.last_sent_response == (
            "Please provide both the day of the week and time of day."
        )

    async def test_on_message_buy_bad_day(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy 100 fun morning"))
        assert channel.last_sent_response == (
            "Please use monday, wednesday, tuesday, etc for the day parameter."
        )

    async def test_on_message_buy_incomplete_time(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy 100 friday pants"))
        assert channel.last_sent_response == (
            "Please use either morning or evening as the time parameter."
        )

    async def test_on_message_buy_no_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy"))
        assert channel.last_sent_response == (
            "Please include buying price after command name."
        )

    async def test_on_message_buy_nonnumeric_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy foot"))
        assert channel.last_sent_response == ("Buying price must be a number.")

    async def test_on_message_buy_nonpositive_price(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!buy 0"))
        assert channel.last_sent_response == ("Buying price must be greater than zero.")

    async def test_on_message_buy(self, client, channel, lines):
        author = someone()
        amount = somebells()
        await client.on_message(MockMessage(author, channel, f"!buy {amount}"))
        assert channel.last_sent_response == (
            f"Logged buying price of {amount} for user {author}."
        )
        assert lines(client.prices_file) == [
            "author,kind,price,timestamp\n",
            f"{author.id},buy,{amount},{NOW}\n",
        ]

    async def test_on_message_help(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!help"))
        for response in channel.all_sent_responses:
            snap(response)

    async def test_on_message_clear(self, client, channel, lines):
        author = someone()
        await client.on_message(MockMessage(author, channel, f"!buy {somebells()}"))
        await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))
        await client.on_message(MockMessage(author, channel, f"!sell {somebells()}"))

        await client.on_message(MockMessage(author, channel, "!clear"))
        assert channel.last_sent_response == (f"**Cleared history for {author}.**")
        assert lines(client.prices_file) == ["author,kind,price,timestamp\n"]

    async def test_on_message_bestsell(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
        await client.on_message(MockMessage(GUY, channel, "!buy 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestsell"))
        assert channel.last_sent_response == (
            "__**Best Selling Prices in the Last 12 Hours**__\n"
            f"> **{BUDDY}:** now for 600 bells\n"
            f"> **{FRIEND}:** now for 200 bells"
        )

    async def test_on_message_bestsell_timezone(self, client, channel):
        friend_tz = "America/Los_Angeles"
        await client.on_message(
            MockMessage(FRIEND, channel, f"!pref timezone {friend_tz}")
        )
        friend_now = NOW.astimezone(pytz.timezone(friend_tz))

        buddy_tz = "Canada/Saskatchewan"
        await client.on_message(MockMessage(BUDDY, channel, f"!pref timezone {buddy_tz}"))
        buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

        guy_tz = "Africa/Abidjan"
        await client.on_message(MockMessage(GUY, channel, f"!pref timezone {guy_tz}"))
        # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 200"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 120"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 600"))
        await client.on_message(MockMessage(GUY, channel, "!buy 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestsell"))
        assert channel.last_sent_response == (
            "__**Best Selling Prices in the Last 12 Hours**__\n"
            f"> **{BUDDY}:** {turbot.h(buddy_now)} for 600 bells\n"
            f"> **{FRIEND}:** {turbot.h(friend_now)} for 200 bells"
        )

    async def test_on_message_oops(self, client, channel, lines):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!oops"))
        assert channel.last_sent_response == (
            f"**Deleting last logged price for {author}.**"
        )
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
        assert channel.last_sent_response == (
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_history_without_name(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!history"))
        assert channel.last_sent_response == (
            f"__**Historical info for {author}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells now\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells now\n"
            f"> Can buy turnips from Daisy Mae for 3 bells now"
        )

    async def test_on_message_history_with_name(self, client, channel):
        await client.on_message(MockMessage(BUDDY, channel, "!buy 1"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 2"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 3"))

        await client.on_message(MockMessage(GUY, channel, f"!history {BUDDY.name}"))
        assert channel.last_sent_response == (
            f"__**Historical info for {BUDDY}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells now\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells now\n"
            f"> Can buy turnips from Daisy Mae for 3 bells now"
        )

    async def test_on_message_history_timezone(self, client, channel):
        author = someone()
        their_tz = "America/Los_Angeles"
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {their_tz}")
        )
        their_now = NOW.astimezone(pytz.timezone(their_tz))

        await client.on_message(MockMessage(author, channel, "!buy 1"))
        await client.on_message(MockMessage(author, channel, "!sell 2"))
        await client.on_message(MockMessage(author, channel, "!buy 3"))

        await client.on_message(MockMessage(author, channel, "!history"))
        assert channel.last_sent_response == (
            f"__**Historical info for {author}**__\n"
            f"> Can buy turnips from Daisy Mae for 1 bells {turbot.h(their_now)}\n"
            f"> Can sell turnips to Timmy & Tommy for 2 bells {turbot.h(their_now)}\n"
            f"> Can buy turnips from Daisy Mae for 3 bells {turbot.h(their_now)}"
        )

    async def test_on_message_bestbuy(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestbuy"))
        assert channel.last_sent_response == (
            "__**Best Buying Prices in the Last 12 Hours**__\n"
            f"> **{BUDDY}:** now for 60 bells\n"
            f"> **{FRIEND}:** now for 100 bells"
        )

    async def test_on_message_bestbuy_timezone(self, client, channel):
        friend_tz = "America/Los_Angeles"
        await client.on_message(
            MockMessage(FRIEND, channel, f"!pref timezone {friend_tz}")
        )
        friend_now = NOW.astimezone(pytz.timezone(friend_tz))

        buddy_tz = "Canada/Saskatchewan"
        await client.on_message(MockMessage(BUDDY, channel, f"!pref timezone {buddy_tz}"))
        buddy_now = NOW.astimezone(pytz.timezone(buddy_tz))

        guy_tz = "Africa/Abidjan"
        await client.on_message(MockMessage(GUY, channel, f"!pref timezone {guy_tz}"))
        # guy_now = NOW.astimezone(pytz.timezone(guy_tz))

        await client.on_message(MockMessage(FRIEND, channel, "!buy 100"))
        await client.on_message(MockMessage(FRIEND, channel, "!sell 600"))
        await client.on_message(MockMessage(BUDDY, channel, "!buy 60"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 90"))
        await client.on_message(MockMessage(BUDDY, channel, "!sell 200"))
        await client.on_message(MockMessage(GUY, channel, "!sell 800"))

        await client.on_message(MockMessage(someone(), channel, "!bestbuy"))
        assert channel.last_sent_response == (
            "__**Best Buying Prices in the Last 12 Hours**__\n"
            f"> **{BUDDY}:** {turbot.h(buddy_now)} for 60 bells\n"
            f"> **{FRIEND}:** {turbot.h(friend_now)} for 100 bells"
        )

    async def test_on_message_graph(self, client, channel, graph):
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

    async def test_on_message_lastweek_none(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!lastweek"))
        assert channel.last_sent_response == ("No graph from last week.")

    async def test_on_message_lastweek_capitalized(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!LASTWEEK"))
        assert channel.last_sent_response == ("No graph from last week.")

    async def test_on_message_lastweek(self, client, channel, lastweek):
        await client.on_message(MockMessage(someturbotadmin(), channel, "!reset"))
        assert channel.last_sent_response == ("**Resetting data for a new week!**")
        lastweek.assert_called_with(channel, None, turbot.LASTWEEKCMD_FILE)
        assert Path(turbot.LASTWEEKCMD_FILE).exists()

        await client.on_message(MockMessage(someone(), channel, "!lastweek"))
        channel.sent.assert_called_with(
            "__**Historical Graph from Last Week**__", file=Matching(is_discord_file)
        )

    async def test_on_message_reset_not_admin(self, client, channel, lines, freezer):
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
        assert channel.last_sent_response == ("User is not a Turbot Admin")
        with open(client.prices_file) as f:
            assert f.readlines() == old_data

        assert not Path(turbot.LASTWEEKCMD_FILE).exists()

    async def test_on_message_reset_admin(
        self, client, channel, lines, freezer, lastweek
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
        assert channel.last_sent_response == ("**Resetting data for a new week!**")
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
        assert channel.last_sent_response == (
            "Please provide the name of something to mark as collected."
        )

    async def test_on_message_collect(self, client, channel, lines):
        # first collect some valid fossils
        author = someone()
        fossils = "amber, ammonite  ,ankylo skull,amber, a foot"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
        assert channel.last_sent_response == (
            "Marked the following fossils as collected:\n"
            "> amber, ammonite, ankylo skull\n"
            "Unrecognized collectable names:\n"
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
        assert channel.last_sent_response == (
            "The following fossils had already been collected:\n"
            "> amber, ammonite, ankylo skull\n"
            "Unrecognized collectable names:\n"
            "> a foot"
        )

        # then collect some more with dupes
        fossils = "amber,an arm,plesio body"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))
        assert channel.last_sent_response == (
            "Marked the following fossils as collected:\n"
            "> plesio body\n"
            "The following fossils had already been collected:\n"
            "> amber\n"
            "Unrecognized collectable names:\n"
            "> an arm"
        )
        assert lines(client.fossils_file) == [f"{author.id},plesio body\n"]

    async def test_on_message_collect_fossils_congrats(self, client, channel):
        everything = sorted(list(turbot.FOSSILS_SET))
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
        assert channel.last_sent_response == (
            "Marked the following fossils as collected:\n"
            f"> {rest_str}\n"
            "**Congratulations, you've collected all fossils!**"
        )

    async def test_on_message_uncollect_art(self, client, channel, lines):
        # first collect some fossils
        author = someone()
        art = "great statue, sinking painting ,academic painting"
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))

        # then delete some of them
        art = "great statue, anime waifu, ancient statue, academic painting"
        await client.on_message(MockMessage(author, channel, f"!uncollect {art}"))
        assert channel.last_sent_response == (
            "Unmarked the following pieces of art as collected:\n"
            "> academic painting, great statue\n"
            "The following pieces of art were already marked as not collected:\n"
            "> ancient statue\n"
            "Unrecognized collectable names:\n"
            "> anime waifu"
        )
        with open(client.art_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},sinking painting\n"]

        # then delete the same ones again
        await client.on_message(MockMessage(author, channel, f"!uncollect {art}"))
        assert channel.last_sent_response == (
            "The following pieces of art were already marked as not collected:\n"
            "> academic painting, ancient statue, great statue\n"
            "Unrecognized collectable names:\n"
            "> anime waifu"
        )
        with open(client.art_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},sinking painting\n"]

        # and delete one more
        await client.on_message(
            MockMessage(author, channel, f"!uncollect sinking painting")
        )
        assert channel.last_sent_response == (
            "Unmarked the following pieces of art as collected:\n" "> sinking painting"
        )
        with open(client.art_file) as f:
            assert f.readlines() == ["author,name\n"]

    async def test_on_message_search_art_no_need_with_bad(self, client, channel):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(BUDDY, channel, "!collect sinking painting, great statue")
        )
        await client.on_message(
            MockMessage(
                GUY, channel, "!collect sinking painting, great statue, wistful painting",
            )
        )

        await client.on_message(
            MockMessage(
                PUNK, channel, "!search sinking painting, great statue, anime waifu"
            )
        )
        assert channel.last_sent_response == (
            "> No one needs: great statue, sinking painting\n"
            "Did not recognize the following collectables:\n"
            "> anime waifu"
        )

    async def test_on_message_search_art(self, client, channel):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
        )
        await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect sinking painting, great statue")
        )

        query = "sinking painting, great statue, wistful painting"
        await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
        channel.last_sent_response == (
            "__**Art Search**__\n"
            f"> {BUDDY} needs: great statue, wistful painting\n"
            f"> {FRIEND} needs: wistful painting\n"
            f"> {GUY} needs: wistful painting"
        )

    async def test_on_message_search_art_with_bad(self, client, channel):
        await client.on_message(
            MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
        )
        await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect sinking painting, great statue")
        )

        query = "sinking painting, great statue, wistful painting, anime waifu"
        await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
        assert channel.last_sent_response == (
            "> No one needs: sinking painting\n"
            f"> {BUDDY} needs arts: great statue, wistful painting\n"
            f"> {FRIEND} needs arts: wistful painting\n"
            f"> {GUY} needs arts: wistful painting\n"
            "Did not recognize the following collectables:\n"
            "> anime waifu"
        )

    async def test_on_message_count_no_params(self, client, channel, lines):
        await client.on_message(MockMessage(someone(), channel, "!count"))
        assert channel.last_sent_response == (
            "Please provide at least one user name to search for."
        )

    async def test_on_message_count_bad_name(self, client, channel, lines):
        await client.on_message(MockMessage(someone(), channel, f"!count {PUNK.name}"))
        assert channel.last_sent_response == (
            f"__**Did not recognize the following names**__\n> {PUNK.name}"
        )

    async def test_on_message_count_when_nothing_collected(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!count {BUDDY.name}"))
        assert channel.last_sent_response == (
            "__**Fossil Count**__\n"
            f"> **{BUDDY}** has 73 fossils remaining.\n"
            "__**Art Count**__\n"
            f"> **{BUDDY}** has 43 pieces of art remaining."
        )

    async def test_on_message_count_art(self, client, channel, snap):
        author = someone()
        await client.on_message(
            MockMessage(FRIEND, channel, "!collect sinking painting, great statue")
        )
        await client.on_message(MockMessage(BUDDY, channel, "!collect sinking painting"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect sinking painting, great statue")
        )

        users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
        await client.on_message(MockMessage(author, channel, f"!count {users}"))
        snap(channel.last_sent_response)

    async def test_on_message_collected_some(self, client, channel):
        author = someone()
        art = "sinking painting, academic painting, great statue"
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))

        fossils = "amber, ammonite, ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(author, channel, "!collected"))
        assert channel.last_sent_response == (
            f"__**3 Pieces of art donated by {author}**__\n"
            ">>> academic painting, great statue, sinking painting\n"
            f"__**3 Fossils donated by {author}**__\n"
            ">>> amber, ammonite, ankylo skull"
        )

    async def test_on_message_collected_all(self, client, channel):
        author = someone()

        all_art = ",".join(turbot.ART_SET)
        await client.on_message(MockMessage(author, channel, f"!collect {all_art}"))

        all_fossils = ",".join(turbot.FOSSILS_SET)
        await client.on_message(MockMessage(author, channel, f"!collect {all_fossils}"))

        await client.on_message(MockMessage(author, channel, f"!collected"))
        assert channel.last_sent_response == (
            "**Congratulations, you've collected all fossils!**\n"
            "**Congratulations, you've collected all art!**"
        )

    async def test_on_message_collected_art_no_name(self, client, channel):
        author = DUDE
        art = "sinking painting, academic painting, great statue"
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))

        await client.on_message(MockMessage(author, channel, "!collected"))
        assert channel.last_sent_response == (
            f"__**3 Pieces of art donated by {DUDE}**__\n"
            ">>> academic painting, great statue, sinking painting"
        )

    async def test_on_message_collected_art_congrats(self, client, channel):
        everything = ",".join(turbot.ART_SET)
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collected"))
        channel.last_sent_response == "**Congratulations, you've collected all art!**"

    async def test_on_message_uncollected_art_congrats(self, client, channel, snap):
        everything = ",".join(turbot.ART_SET)
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {everything}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!uncollected"))
        snap(channel.last_sent_response)

    async def test_on_message_collected_art_with_name(self, client, channel):
        art = "sinking painting, academic painting, great statue"
        await client.on_message(MockMessage(GUY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
        assert channel.last_sent_response == (
            f"__**3 Pieces of art donated by {GUY}**__\n"
            ">>> academic painting, great statue, sinking painting"
        )

    async def test_on_message_collected_art_bad_name(self, client, channel):
        await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
        assert channel.last_sent_response == (
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_collect_fish(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!collect bitterling"))
        assert channel.last_sent_response == "Collecting fish is not supported yet."

    async def test_on_message_collect_bugs(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!collect mantis"))
        assert channel.last_sent_response == "Collecting bugs is not supported yet."

    async def test_on_message_uncollect_fish(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!uncollect bitterling"))
        assert channel.last_sent_response == "Uncollecting fish is not supported yet."

    async def test_on_message_uncollect_bugs(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!uncollect mantis"))
        assert channel.last_sent_response == "Uncollecting bugs is not supported yet."

    async def test_on_message_collect_art(self, client, channel, lines):
        # first collect some art
        author = BUDDY
        art = "academic painting, sinking painting, anime waifu"
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))
        assert channel.last_sent_response == (
            "Marked the following art as collected:\n"
            "> academic painting, sinking painting\n"
            "Unrecognized collectable names:\n"
            "> anime waifu"
        )
        assert set(lines(client.art_file)) == {
            "author,name\n",
            f"{author.id},academic painting\n",
            f"{author.id},sinking painting\n",
        }

        # collect them again
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))
        assert channel.last_sent_response == (
            "The following art had already been collected:\n"
            "> academic painting, sinking painting\n"
            "Unrecognized collectable names:\n"
            "> anime waifu"
        )

        # collect some new stuff, but with some dupes
        art = "body pillow, sinking painting, tremendous statue"
        await client.on_message(MockMessage(author, channel, f"!collect {art}"))
        assert channel.last_sent_response == (
            "Marked the following art as collected:\n"
            "> tremendous statue\n"
            "The following art had already been collected:\n"
            "> sinking painting\n"
            "Unrecognized collectable names:\n"
            "> body pillow"
        )

        assert lines(client.art_file) == [f"{author.id},tremendous statue\n"]

    async def test_on_message_collect_art_congrats(self, client, channel, snap):
        everything = sorted(list(turbot.ART.name.unique()))
        some, rest = everything[:10], everything[10:]

        # someone else collects some pieces
        art = "academic painting, sinking painting, tremendous statue"
        await client.on_message(MockMessage(GUY, channel, f"!collect {art}"))

        # Buddy collects some
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collect {', '.join(some)}")
        )

        # Friend collects a different set
        art = "mysterious painting, twinkling painting"
        await client.on_message(MockMessage(FRIEND, channel, f"!collect {art}"))

        # Buddy collects the rest
        await client.on_message(
            MockMessage(BUDDY, channel, f"!collect {', '.join(rest)}")
        )
        snap(channel.last_sent_response)

    async def test_on_message_collected_bad_name(self, client, channel):
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
        assert channel.last_sent_response == (
            "Can not find the user named punk in this channel."
        )

    async def test_on_message_collected_no_name(self, client, channel):
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collected"))
        assert channel.last_sent_response == (
            f"__**2 Pieces of art donated by {BUDDY}**__\n"
            ">>> academic painting, sinking painting"
        )

    async def test_on_message_uncollected_bad_name(self, client, channel):
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!uncollected {PUNK.name}"))
        assert channel.last_sent_response == (
            "Can not find the user named punk in this channel."
        )

    async def test_on_message_uncollected_no_name(self, client, channel, snap):
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!uncollected"))
        snap(channel.last_sent_response)

    async def test_on_message_uncollected_with_name(self, client, channel, snap):
        art = "academic painting, sinking painting"
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {art}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!uncollected {BUDDY.name}"))
        snap(channel.last_sent_response)

    async def test_on_message_search_no_list(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!search"))
        assert channel.last_sent_response == (
            "Please provide the name of a collectable to search for."
        )

    async def test_on_message_search_all_no_need(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
        )

        await client.on_message(MockMessage(PUNK, channel, "!search amber, ammonite"))
        assert channel.last_sent_response == ("No one currently needs this.")

    async def test_on_message_search_fossil_no_need_with_bad(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber, ammonite"))
        await client.on_message(
            MockMessage(GUY, channel, "!collect amber, ammonite, coprolite")
        )

        await client.on_message(
            MockMessage(PUNK, channel, "!search amber, ammonite, unicorn bits")
        )
        assert channel.last_sent_response == (
            "> No one needs: amber, ammonite\n"
            "Did not recognize the following collectables:\n"
            "> unicorn bits"
        )

    async def test_on_message_search_fossil(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        query = "amber, ammonite, ankylo skull"
        await client.on_message(MockMessage(PUNK, channel, f"!search {query}"))
        assert channel.last_sent_response == (
            "> No one needs: amber\n"
            f"> {BUDDY} needs fossils: ammonite, ankylo skull\n"
            f"> {FRIEND} needs fossils: ankylo skull\n"
            f"> {GUY} needs fossils: ankylo skull"
        )

    async def test_on_message_search_fossil_with_bad(self, client, channel):
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        await client.on_message(
            MockMessage(
                PUNK, channel, "!search amber, ammonite, ankylo skull, unicorn bits"
            )
        )
        assert channel.last_sent_response == (
            "> No one needs: amber\n"
            f"> {BUDDY} needs fossils: ammonite, ankylo skull\n"
            f"> {FRIEND} needs fossils: ankylo skull\n"
            f"> {GUY} needs fossils: ankylo skull\n"
            "Did not recognize the following collectables:\n"
            "> unicorn bits"
        )

    async def test_on_message_search_with_only_bad(self, client, channel):
        await client.on_message(MockMessage(PUNK, channel, "!search unicorn bits"))
        assert channel.last_sent_response == (
            "Did not recognize the following collectables:\n" "> unicorn bits"
        )

    async def test_on_message_uncollect_no_list(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!uncollect"))
        assert channel.last_sent_response == (
            "Please provide the name of something to mark as uncollected."
        )

    async def test_on_message_uncollect_fossil(self, client, channel):
        # first collect some fossils
        author = someone()
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        # then delete some of them
        fossils = "amber, a foot, coprolite, ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!uncollect {fossils}"))
        assert channel.last_sent_response == (
            "Unmarked the following fossils as collected:\n"
            "> amber, ankylo skull\n"
            "The following fossils were already marked as not collected:\n"
            "> coprolite\n"
            "Unrecognized collectable names:\n"
            "> a foot"
        )
        with open(client.fossils_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},ammonite\n"]

        # delete the same ones again
        await client.on_message(MockMessage(author, channel, f"!uncollect {fossils}"))
        assert channel.last_sent_response == (
            "The following fossils were already marked as not collected:\n"
            "> amber, ankylo skull, coprolite\n"
            "Unrecognized collectable names:\n"
            "> a foot"
        )
        with open(client.fossils_file) as f:
            assert f.readlines() == ["author,name\n", f"{author.id},ammonite\n"]

        # and delete one more
        await client.on_message(MockMessage(author, channel, f"!uncollect ammonite"))
        assert channel.last_sent_response == (
            "Unmarked the following fossils as collected:\n> ammonite"
        )
        with open(client.fossils_file) as f:
            assert f.readlines() == ["author,name\n"]

    async def test_on_message_uncollect_with_only_bad(self, client, channel):
        fossils = "a foot, unicorn bits"
        await client.on_message(MockMessage(someone(), channel, f"!uncollect {fossils}"))
        assert channel.last_sent_response == (
            "Unrecognized collectable names:\n> a foot, unicorn bits"
        )

    async def test_on_message_allfossils(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, "!allfossils"))
        snap(channel.last_sent_response)

    async def test_on_message_collected_fossils_congrats(self, client, channel):
        author = someone()
        everything = ", ".join(sorted(turbot.FOSSILS_SET))
        await client.on_message(MockMessage(author, channel, f"!collect {everything}"))

        await client.on_message(MockMessage(author, channel, "!collected"))
        assert channel.last_sent_response == (
            "**Congratulations, you've collected all fossils!**"
        )

    async def test_on_message_uncollected_fossils_congrats(self, client, channel, snap):
        author = DUDE
        everything = ", ".join(sorted(turbot.FOSSILS_SET))
        await client.on_message(MockMessage(author, channel, f"!collect {everything}"))

        await client.on_message(MockMessage(author, channel, "!uncollected"))
        snap(channel.last_sent_response)

    async def test_on_message_neededfossils(self, client, channel):
        everything = sorted(list(turbot.FOSSILS_SET))

        fossils = ",".join(everything[3:])
        await client.on_message(MockMessage(BUDDY, channel, f"!collect {fossils}"))

        fossils = ",".join(everything[20:])
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        fossils = ",".join(everything)
        await client.on_message(MockMessage(FRIEND, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(someone(), channel, "!neededfossils"))
        assert channel.last_sent_response == (
            f"> **{BUDDY}** needs acanthostega, amber, ammonite\n"
            f"> **{GUY}** needs _more than 10 fossils..._"
        )

    async def test_on_message_neededfossils_none(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!neededfossils"))
        assert channel.last_sent_response == (
            "No fossils are known to be needed at this time, "
            "new users must collect at least one fossil."
        )

    async def test_on_message_collected_fossils_no_name(self, client, channel):
        author = someone()
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(author, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(author, channel, "!collected"))
        assert channel.last_sent_response == (
            f"__**3 Fossils donated by {author}**__\n" ">>> amber, ammonite, ankylo skull"
        )

    async def test_on_message_collected_fossils_with_name(self, client, channel):
        fossils = "amber, ammonite ,ankylo skull"
        await client.on_message(MockMessage(GUY, channel, f"!collect {fossils}"))

        await client.on_message(MockMessage(BUDDY, channel, f"!collected {GUY.name}"))
        assert channel.last_sent_response == (
            f"__**3 Fossils donated by {GUY}**__\n" ">>> amber, ammonite, ankylo skull"
        )

    async def test_on_message_collected_fossils_bad_name(self, client, channel):
        await client.on_message(MockMessage(BUDDY, channel, f"!collected {PUNK.name}"))
        assert channel.last_sent_response == (
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_count_fossils(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(FRIEND, channel, "!collect amber, ammonite"))
        await client.on_message(MockMessage(BUDDY, channel, "!collect amber"))
        await client.on_message(MockMessage(GUY, channel, "!collect amber, ammonite"))

        users = ", ".join([FRIEND.name, BUDDY.name, GUY.name, PUNK.name])
        await client.on_message(MockMessage(author, channel, f"!count {users}"))
        assert channel.last_sent_response == (
            "__**Fossil Count**__\n"
            f"> **{BUDDY}** has 72 fossils remaining.\n"
            f"> **{FRIEND}** has 71 fossils remaining.\n"
            f"> **{GUY}** has 71 fossils remaining.\n"
            "__**Art Count**__\n"
            f"> **{BUDDY}** has 43 pieces of art remaining.\n"
            f"> **{FRIEND}** has 43 pieces of art remaining.\n"
            f"> **{GUY}** has 43 pieces of art remaining.\n"
            "__**Did not recognize the following names**__\n"
            f"> {PUNK.name}"
        )

    async def test_on_message_predict_no_buy(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!predict"))
        assert channel.last_sent_response == (
            f"There is no recent buy price for {author}."
        )

    async def test_on_message_predict_bad_user(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!predict {PUNK.name}"))
        assert channel.last_sent_response == (
            f"Can not find the user named {PUNK.name} in this channel."
        )

    async def test_on_message_predict(self, client, channel, freezer):
        author = someone()

        sunday_am = datetime(2020, 4, 26, 9, tzinfo=pytz.utc)
        freezer.move_to(sunday_am)
        await client.on_message(MockMessage(author, channel, "!buy 110"))

        freezer.move_to(sunday_am + timedelta(days=1))
        await client.on_message(MockMessage(author, channel, "!sell 100"))
        freezer.move_to(sunday_am + timedelta(days=1, hours=12))
        await client.on_message(MockMessage(author, channel, "!sell 95"))

        freezer.move_to(sunday_am + timedelta(days=2))
        await client.on_message(MockMessage(author, channel, "!sell 90"))
        freezer.move_to(sunday_am + timedelta(days=2, hours=12))
        await client.on_message(MockMessage(author, channel, "!sell 85"))

        freezer.move_to(sunday_am + timedelta(days=4))
        await client.on_message(MockMessage(author, channel, "!sell 90"))

        freezer.move_to(sunday_am + timedelta(days=5))
        await client.on_message(MockMessage(author, channel, "!sell 120"))

        await client.on_message(MockMessage(author, channel, "!predict"))
        channel.sent.assert_called_with(
            f"__**Predictive Graph for {author}**__\n"
            "Details: <https://turnipprophet.io/?prices=110.100.95.90.85...90..120>",
            file=Matching(is_discord_file),
        )

    async def test_on_message_predict_with_timezone(self, client, channel, freezer):
        author = someone()
        user_tz = pytz.timezone("America/Los_Angeles")
        await client.on_message(
            MockMessage(author, channel, f"!pref timezone {user_tz.zone}")
        )

        # sunday morning buy
        sunday_morning = datetime(year=2020, month=4, day=19, hour=6, tzinfo=user_tz)
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
            f"__**Predictive Graph for {author}**__\n"
            "Details: <https://turnipprophet.io/?prices=110.87.72>",
            file=Matching(is_discord_file),
        )

    async def test_get_last_price(self, client, channel, freezer):
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

    async def test_on_message_fish_no_hemisphere(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!fish"))
        assert channel.last_sent_response == (
            "Please enter your hemisphere choice first "
            "using the !pref hemisphere command."
        )

    async def test_on_message_fish_none_found(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))

        await client.on_message(MockMessage(author, channel, "!fish Blinky"))
        assert channel.last_sent_response == (
            'Did not find any fish searching for "Blinky".'
        )

    async def test_on_message_fish_multiple_users(self, client, channel):
        await client.on_message(MockMessage(GUY, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(FRIEND, channel, "!pref hemisphere northern"))

        await client.on_message(MockMessage(GUY, channel, "!fish sea"))
        await client.on_message(MockMessage(BUDDY, channel, "!fish sea"))
        await client.on_message(MockMessage(FRIEND, channel, "!fish sea"))

    async def test_on_message_fish_search_query(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish ch"))
        snap(channel.last_sent_response)

    async def test_on_message_fish_search_leaving(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish leaving"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_fish_search_arriving(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish arriving"))
        snap(channel.last_sent_response)

    async def test_on_message_fish(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish"))
        snap(channel.last_sent_response)

    async def test_on_message_fish_case_insensitive(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!fish SeA"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_bugs_case_insensitive(self, client, channel, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs TaRaNtUlA"))
        snap(channel.all_sent_embeds_json)

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
        assert channel.last_sent_response == (
            "Please enter your hemisphere choice "
            "first using the !pref hemisphere command."
        )

    async def test_on_message_bug_none_found(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs Shelob"))
        assert channel.last_sent_response == (
            'Did not find any bugs searching for "Shelob".'
        )

    async def test_on_message_bug_multiple_users(self, client, channel):
        await client.on_message(MockMessage(GUY, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(BUDDY, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(FRIEND, channel, "!pref hemisphere northern"))

        await client.on_message(MockMessage(GUY, channel, "!bugs butt"))
        await client.on_message(MockMessage(BUDDY, channel, "!bugs butt"))
        await client.on_message(MockMessage(FRIEND, channel, "!bugs butt"))

    async def test_on_message_bug_search_query_many(
        self, client, channel, without_bugs_header, snap
    ):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs butt"))
        snap(channel.last_sent_response)

    async def test_on_message_bug_search_query_few(
        self, client, channel, without_bugs_header, snap
    ):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs beet"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_bug_header(self, client, channel, with_bugs_header, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs butt"))
        snap(channel.last_sent_response)

    async def test_on_message_bug_search_leaving(
        self, client, channel, without_bugs_header, snap
    ):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs leaving"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_bug_search_arriving(
        self, client, channel, without_bugs_header, snap
    ):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs arriving"))
        snap(channel.last_sent_response)

    async def test_on_message_new(self, client, channel, without_bugs_header, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!new"))
        snap(channel.all_sent_responses[1])
        snap(channel.all_sent_responses[2])
        assert len(channel.all_sent_responses) == 3

    async def test_on_message_new_first_day(
        self, client, channel, freezer, without_bugs_header, snap
    ):
        first_day_of_the_month = datetime(2020, 5, 1, tzinfo=pytz.utc)
        freezer.move_to(first_day_of_the_month)

        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!new"))
        snap(channel.all_sent_responses[1])
        snap(channel.all_sent_responses[2])
        assert len(channel.all_sent_responses) == 3

    async def test_on_message_bug(self, client, channel, without_bugs_header, snap):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(author, channel, "!bugs"))
        snap(channel.all_sent_responses[1])
        snap(channel.all_sent_responses[2])
        assert len(channel.all_sent_responses) == 3

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

        with open(Path(DAT_ROOT) / "quotes.txt") as f:
            text = f.read()
            pages = subject(text)
            assert len(pages) == 2
            assert all(len(page) <= 2000 for page in pages)
            assert pages == [text[0:1926], text[1926:]]

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

    async def test_on_message_info(self, client, channel, snap):
        author = DUDE
        prefs = {
            "hemisphere": "norTHErn",
            "timezone": "America/Los_Angeles",
            "island": "Kriti",
            "friend": "Sw-1111----2222-3333",
            "fruit": "pEaCh",
            "nickname": "Phèdre nó Delaunay de Montrève",
            "creator": "ma---  4444----555 5-6666--",
        }
        for pref, value in prefs.items():
            await client.on_message(MockMessage(author, channel, f"!pref {pref} {value}"))
            await client.on_message(MockMessage(author, channel, f"!info {author.name}"))
        snap(channel.all_sent_embeds_json)

    async def test_get_user_prefs(self, client, channel, snap):
        author = DUDE
        prefs = {
            "hemisphere": "norTHErn",
            "timezone": "America/Los_Angeles",
            "island": "Kriti",
            "friend": "Sw-1111----2222-3333",
            # "fruit": Not Set,
            "nickname": "Phèdre nó Delaunay de Montrève",
            "creator": "ma---  4444----555 5-6666--",
        }
        for pref, value in prefs.items():
            await client.on_message(MockMessage(author, channel, f"!pref {pref} {value}"))

        assert client.get_user_prefs(author.id) == {
            "friend": "111122223333",
            "hemisphere": "northern",
            "island": "Kriti",
            "nickname": "Phèdre nó Delaunay de Montrève",
            "timezone": pytz.timezone(prefs["timezone"]),
            "creator": "444455556666",
        }

        # unload in-memory users data
        client._users_data = None

        assert client.get_user_prefs(author.id) == {
            "friend": "111122223333",
            "hemisphere": "northern",
            "island": "Kriti",
            "nickname": "Phèdre nó Delaunay de Montrève",
            "timezone": pytz.timezone(prefs["timezone"]),
            "creator": "444455556666",
        }

        assert client.get_user_prefs(PUNK.id) == {}

    async def test_get_user_prefs_friend_code(self, client, channel, snap):
        author = someone()
        with open(client.users_file, "w") as f:
            f.writelines(
                [
                    "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                    f"{author.id},,,,111122223333.0,,,\n",
                ]
            )
        assert client.get_user_prefs(author.id) == {"friend": "111122223333.0"}
        await client.on_message(MockMessage(author, channel, f"!info {author.name}"))
        assert (
            next(
                f["value"]
                for f in channel.last_sent_embed["fields"]
                if f["name"] == "Friend code"
            )
            == "SW-1111-2222-3333"
        )

    async def test_on_message_about(self, client, channel, snap):
        await client.on_message(MockMessage(someone(), channel, f"!about"))
        snap(channel.all_sent_embeds_json)

    async def test_on_message_info_old_user(self, client, channel, monkeypatch):
        # Simulate the condition where a user exists in the data file,
        # but is no longer on the server.
        monkeypatch.setattr(turbot, "discord_user_name", lambda *_: None)

        await client.on_message(MockMessage(DUDE, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
        assert channel.last_sent_response == "No users found."

    async def test_on_message_info_not_found(self, client, channel):
        await client.on_message(MockMessage(DUDE, channel, "!pref hemisphere northern"))
        await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
        assert channel.last_sent_response == "No users found."

    async def test_on_message_info_no_users(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!info {PUNK.name}"))
        assert channel.last_sent_response == "No users found."

    async def test_on_message_info_no_prefs(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, f"!buy 100"))
        await client.on_message(MockMessage(someone(), channel, f"!info {author.name}"))
        assert channel.last_sent_response == f"> **{author}** has no preferences."

    async def test_on_message_info_no_params(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!info"))
        assert channel.last_sent_response == "Please provide a search term."

    async def test_on_message_search_bugs(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!search mantis"))
        channel.last_sent_response == "Searching for bugs is not supported yet."

    async def test_on_message_search_fish(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, f"!search bitterling"))
        channel.last_sent_response == "Searching for fish is not supported yet."

    async def test_discord_user_from_name_guard(self, channel):
        assert turbot.discord_user_from_name(channel, None) == None

    async def test_discord_user_name_guard(self, channel):
        assert turbot.discord_user_name(channel, None) == None

    async def test_on_message_pref_no_params(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref"))
        assert channel.last_sent_response == (
            "Please provide a preference and a value, possible preferences include "
            "hemisphere, timezone, island, friend, fruit, nickname, creator."
        )

    async def test_on_message_pref_no_value(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref creator"))
        assert channel.last_sent_response == (
            "Please provide the value for your creator preference."
        )

    async def test_on_message_pref_invalid_pref(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref shazbot"))
        assert channel.last_sent_response == (
            "Please provide a valid preference name, possible preferences include "
            "hemisphere, timezone, island, friend, fruit, nickname, creator."
        )

    async def test_on_message_pref_hemisphere_invalid(self, client, channel):
        await client.on_message(
            MockMessage(someone(), channel, "!pref hemisphere upwards")
        )
        assert channel.last_sent_response == (
            'Please provide either "northern" or "southern" as your hemisphere name.'
        )

    async def test_on_message_pref_hemisphere(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, "!pref hemisphere souTherN"))
        assert channel.last_sent_response == (
            f"Registered hemisphere preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},southern,,,,,,\n",
            ]

        await client.on_message(MockMessage(author, channel, "!pref hemisphere NoRthErn"))
        assert channel.last_sent_response == (
            f"Registered hemisphere preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},northern,,,,,,\n",
            ]

    async def test_on_message_pref_friend_invalid(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref friend upwards"))
        assert channel.last_sent_response == (
            "Your switch friend code should be 12 numbers."
        )

    async def test_on_message_pref_friend(self, client, channel):
        author = someone()
        await client.on_message(
            MockMessage(author, channel, f"!pref friend sw-1234-5678-9012")
        )
        assert channel.last_sent_response == f"Registered friend preference for {author}."
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,,,123456789012,,,\n",
            ]

    async def test_on_message_pref_creator_invalid(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref creator upwards"))
        assert channel.last_sent_response == (
            "Your Animal Crossing creator code should be 12 numbers."
        )

    async def test_on_message_pref_creator(self, client, channel):
        author = someone()
        await client.on_message(
            MockMessage(author, channel, f"!pref creator mA-1234-5678-9012")
        )
        assert (
            channel.last_sent_response == f"Registered creator preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,,,,,,123456789012\n",
            ]

    async def test_on_message_pref_fruit_invalid(self, client, channel):
        await client.on_message(MockMessage(someone(), channel, "!pref fruit upwards"))
        assert channel.last_sent_response == (
            "Your native fruit can be apple, cherry, orange, peach, or pear."
        )

    async def test_on_message_pref_fruit(self, client, channel):
        author = someone()
        await client.on_message(MockMessage(author, channel, f"!pref fruit apple"))
        assert channel.last_sent_response == (
            f"Registered fruit preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,,,,apple,,\n",
            ]

    async def test_on_message_pref_timezone_invalid(self, client, channel):
        await client.on_message(
            MockMessage(someone(), channel, "!pref timezone Mars/Noctis_City")
        )
        assert channel.last_sent_response == (
            "Please provide a valid timezone name, see "
            "https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for the "
            "complete list of TZ names."
        )

    async def test_on_message_pref_timezone(self, client, channel):
        author = someone()
        await client.on_message(
            MockMessage(author, channel, "!pref timezone America/Los_Angeles")
        )
        assert channel.last_sent_response == (
            f"Registered timezone preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,America/Los_Angeles,,,,,\n",
            ]

        await client.on_message(
            MockMessage(author, channel, "!pref timezone Canada/Saskatchewan")
        )
        assert channel.last_sent_response == (
            f"Registered timezone preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,Canada/Saskatchewan,,,,,\n",
            ]

    async def test_on_message_pref_island(self, client, channel):
        author = someone()
        island = "Koholint Island"
        await client.on_message(MockMessage(author, channel, f"!pref island {island}"))
        assert channel.last_sent_response == (
            f"Registered island preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,,{island},,,,\n",
            ]

    async def test_on_message_pref_nickname(self, client, channel):
        author = someone()
        name = "Chuck Noland"
        await client.on_message(MockMessage(author, channel, f"!pref nickname {name}"))
        assert channel.last_sent_response == (
            f"Registered nickname preference for {author}."
        )
        with open(client.users_file) as f:
            assert f.readlines() == [
                "author,hemisphere,timezone,island,friend,fruit,nickname,creator\n",
                f"{author.id},,,,,,{name},\n",
            ]


class TestFigures:
    # Some realistic price data sampled from the wild.
    PRICES = [
        "author,kind,price,timestamp\n",
        f"{FRIEND.id},buy,103,2020-04-05 09:00:00+00:00\n",  # Sunday_AM
        f"{FRIEND.id},sell,112,2020-04-06 09:00:00+00:00\n",  # Monday_AM
        f"{FRIEND.id},sell,116,2020-04-06 13:00:00+00:00\n",  # Monday_PM
        f"{FRIEND.id},sell,80,2020-04-07 09:00:00+00:00\n",  # Tuesday_AM
        # f"{FRIEND.id},sell,None,2020-04-07 13:00:00+00:00\n",  # Tuesday_PM
        f"{FRIEND.id},sell,100,2020-04-08 09:00:00+00:00\n",  # Wednesday_AM
        # f"{FRIEND.id},sell,None,2020-04-08 13:00:00+00:00\n",  # Wednesday_PM
        f"{FRIEND.id},sell,95,2020-04-09 09:00:00+00:00\n",  # Thursday_AM
        # f"{FRIEND.id},sell,None,2020-04-09 13:00:00+00:00\n",  # Thursday_PM
        f"{FRIEND.id},sell,80,2020-04-10 09:00:00+00:00\n",  # Friday_AM
        # f"{FRIEND.id},sell,None,2020-04-10 13:00:00+00:00\n",  # Friday_PM
        # f"{FRIEND.id},sell,None,2020-04-11 09:00:00+00:00\n",  # Saturday_AM
        # f"{FRIEND.id},sell,None,2020-04-11 13:00:00+00:00\n",  # Saturday_PM
        f"{DUDE.id},buy,98,2020-04-05 09:00:00+00:00\n",  # Sunday_AM
        f"{DUDE.id},sell,88,2020-04-06 09:00:00+00:00\n",  # Monday_AM
        f"{DUDE.id},sell,84,2020-04-06 13:00:00+00:00\n",  # Monday_PM
        f"{DUDE.id},sell,81,2020-04-07 09:00:00+00:00\n",  # Tuesday_AM
        f"{DUDE.id},sell,76,2020-04-07 13:00:00+00:00\n",  # Tuesday_PM
        # f"{DUDE.id},sell,None,2020-04-08 09:00:00+00:00\n",# Wednesday_AM
        # f"{DUDE.id},sell,None,2020-04-08 13:00:00+00:00\n",# Wednesday_PM
        f"{DUDE.id},sell,138,2020-04-09 09:00:00+00:00\n",  # Thursday_AM
        f"{DUDE.id},sell,336,2020-04-09 13:00:00+00:00\n",  # Thursday_PM
        f"{DUDE.id},sell,191,2020-04-10 09:00:00+00:00\n",  # Friday_AM
        f"{DUDE.id},sell,108,2020-04-10 13:00:00+00:00\n",  # Friday_PM
        # f"{DUDE.id},sell,None,2020-04-11 09:00:00+00:00\n",  # Saturday_AM
        # f"{DUDE.id},sell,None,2020-04-11 13:00:00+00:00\n",  # Saturday_PM
    ]

    def set_example_prices(self, client):
        with open(client.prices_file, "w") as f:
            f.writelines(self.PRICES)

    def set_bogus_prices(self, client):
        with open(client.prices_file, "w") as f:
            f.writelines(
                [
                    "author,kind,price,timestamp\n",
                    # user that's not in the channel:
                    f"{PUNK.id},buy,100,1982-04-24 01:00:00+00:00\n",
                    # some actually valid data:
                    f"{FRIEND.id},buy,103,2020-04-05 09:00:00+00:00\n",
                    f"{FRIEND.id},sell,112,2020-04-06 09:00:00+00:00\n",
                    # a user with only buy data, no sell data:
                    f"{DUDE.id},buy,98,2020-04-05 09:00:00+00:00\n",
                ]
            )

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
        assert proc.returncode == 0, f"black issues:\n{proc.stdout.decode('utf-8')}"

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
        config_keys = set(turbot.STRINGS.keys())
        assert config_keys - used_keys == set()

    # Tracks the usage of snapshot files over the entire test session.
    # When it fails it means you probably need to clear out any unused snapshot files.
    def test_snapshots(self):
        """Checks that all of the snapshots files are being used."""
        snapshots_dir = REPO_ROOT / "tests" / "snapshots"
        snapshot_files = set(f.name for f in snapshots_dir.glob("*.txt"))
        assert snapshot_files == SNAPSHOTS_USED
