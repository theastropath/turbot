from os.path import dirname, realpath
from pathlib import Path

import alembic
import alembic.config
import pandas as pd
import pytz
from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

PACKAGE_ROOT = Path(dirname(realpath(__file__)))
ASSETS_DIR = PACKAGE_ROOT / "assets"
ALEMBIC_INI = ASSETS_DIR / "alembic.ini"
VERSIONS_DIR = PACKAGE_ROOT / "versions"


Base = declarative_base()


class Fossil(Base):
    __tablename__ = "fossils"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    name = Column(String(50), nullable=False)


class Art(Base):
    __tablename__ = "art"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    name = Column(String(50), nullable=False)


class Fish(Base):
    __tablename__ = "fish"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    name = Column(String(50), nullable=False)


class Bug(Base):
    __tablename__ = "bugs"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    name = Column(String(50), nullable=False)


class Song(Base):
    __tablename__ = "songs"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    name = Column(String(50), nullable=False)


class Price(Base):
    __tablename__ = "prices"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    author = Column(BigInteger, ForeignKey("users.author"), nullable=False)
    kind = Column(String(20), nullable=False)
    price = Column(Integer, nullable=False)
    timestamp = Column(String(50), nullable=False)


class User(Base):
    __tablename__ = "users"
    author = Column(BigInteger, primary_key=True, nullable=False)
    hemisphere = Column(String(15))
    hemisphere = Column(String(50))
    timezone = Column(String(50))
    island = Column(String(50))
    friend = Column(String(20))
    fruit = Column(String(10))
    nickname = Column(String(50))
    creator = Column(String(20))

    prices = relationship("Price", backref="user")
    songs = relationship("Song", backref="user")
    bugs = relationship("Bug", backref="user")
    fish = relationship("Fish", backref="user")
    art = relationship("Art", backref="user")
    fossils = relationship("Fossil", backref="user")

    def get_timezone(self):
        return pytz.timezone(self.timezone) if self.timezone else pytz.UTC


class AuthorizedChannel(Base):
    __tablename__ = "authorized_channels"
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    guild = Column(BigInteger, primary_key=True, nullable=False)
    name = Column(String(100), nullable=False)


def create_all(connection, db_url):
    config = alembic.config.Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(VERSIONS_DIR))
    config.set_main_option("sqlalchemy.url", db_url)
    config.attributes["connection"] = connection
    alembic.command.upgrade(config, "head")


def reverse_all(connection, db_url):
    config = alembic.config.Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(VERSIONS_DIR))
    config.set_main_option("sqlalchemy.url", db_url)
    config.attributes["connection"] = connection
    alembic.command.downgrade(config, "base")


class Data:
    """Persistent and in-memory store for user data."""

    def __init__(self, db_url):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.conn = self.engine.connect()
        create_all(self.conn, db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = Base.metadata
        self.data_types = self.metadata.tables.keys()
        self.columns = {
            data_type: list(
                filter(
                    lambda name: name != "id",
                    (column.name for column in self.metadata.tables[data_type].columns),
                )
            )
            for data_type in self.data_types
        }
        self.dtypes = {
            data_type: {
                column: (
                    "int64"
                    if column in ["author", "guild", "price"]
                    else "datetime64[ns, UTC]"
                    if column == "timestamp"
                    else "str"
                )
                for column in self.columns[data_type]
            }
            for data_type in self.data_types
        }
        self.models = {}
        for cls in Base._decl_class_registry.values():
            if hasattr(cls, "__tablename__") and cls.__tablename__ in self.columns:
                self.models[cls.__tablename__] = cls

    def __getattr__(self, attr):
        if attr not in self.data_types:
            raise RuntimeError(f"there is no data store for {attr}")

        columns = self.columns[attr]
        parse_dates = "timestamp" in columns
        query = f"SELECT {','.join(columns)} FROM {attr};"
        df = pd.read_sql_query(query, self.conn, parse_dates=parse_dates)
        return df.fillna("").astype(self.dtypes[attr])
