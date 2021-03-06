import hashlib
import io
import os

import numpy as np
import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy import Column, DateTime, Float, Integer, LargeBinary, String
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator


engine = None
Session = None


Base = sqlalchemy.orm.declarative_base()


class NumpyArray(TypeDecorator):
    """A SQLAlchemy data type for numpy arrays."""
    # https://docs.sqlalchemy.org/en/13/core/custom_types.html#marshal-json-strings

    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        del dialect
        if value is not None:
            buffer = io.BytesIO()
            np.save(buffer, value)
        return buffer.getvalue()

    def process_result_value(self, value, dialect):
        del dialect
        if value is not None:
            buffer = io.BytesIO(value)
            value = np.load(buffer)
        return value


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True)
    file_hash = Column(String)

    device_manufacturer = Column(String)
    device_model = Column(String)

    datetime_start = Column(DateTime)
    datetime_end = Column(DateTime)

    name = Column(String)
    sport = Column(String)
    sub_sport = Column(String)
    workout = Column(String)

    # One-to-one.
    # https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#one-to-one
    summary = relationship("Summary", back_populates="activity", uselist=False)

    # One-to-many: One activity, several recordings.
    # https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html?highlight=relationship#one-to-many
    recordings = relationship("Recording")


class Summary(Base):
    """Global scalar statistics for an activity."""

    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"))

    # Seconds.
    duration = Column(Integer)

    # Metres.
    distance = Column(Integer)
    ascent = Column(Integer)
    descent = Column(Integer)

    # Median values.
    # Beats per minute (bpm).
    heart_rate = Column(Integer)
    # Steps per minute (spm).
    step_rate = Column(Integer)
    # Metres/second.
    speed = Column(Float)

    # One-to-one.
    # https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#one-to-one
    activity = relationship("Activity", back_populates="summary")


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    array = Column(NumpyArray)
    activity_id = Column(Integer, ForeignKey("activities.id"))

    # One-to-many.
    # https://docs.sqlalchemy.org/en/14/orm/basic_relationships.html#one-to-many
    activity = relationship("Activity", back_populates="recordings")


def make_engine(path: str = "activities.db"):
    global engine
    url: str = f"sqlite:///{path}"
    engine = sqlalchemy.create_engine(url)
    return engine


def make_session():
    global engine, Session
    assert engine is not None, "call make_engine() first"
    if Session is None:
        Session = sqlalchemy.orm.sessionmaker(bind=engine)
    return Session()


def create(engine):
    Base.metadata.create_all(engine)


def hash_file(path: os.PathLike) -> str:
    digest_size = 16
    hasher = hashlib.blake2b(digest_size=digest_size)
    with open(path, "rb") as file_:
        hasher.update(file_.read())
    hexdigest = hasher.hexdigest()
    return hexdigest


def has_activity(path: os.PathLike) -> bool:
    file_hash = hash_file(path)
    query = select(Activity).where(Activity.file_hash == file_hash)
    _ = make_engine()
    session = make_session()
    result = session.execute(query).first()
    return result is not None
