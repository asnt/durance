import os

import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import select
from sqlalchemy import Column, DateTime, Integer, String


engine = None
Session = None


Base = sqlalchemy.orm.declarative_base()


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

    duration = Column(Integer)
    distance = Column(Integer)

    heartrate_mean = Column(Integer)
    heartrate_median = Column(Integer)


def make_engine(path="activities.db"):
    global engine
    url = f"sqlite:///{path}"
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


def hash_file(path: os.PathLike) -> bool:
    import hashlib
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
