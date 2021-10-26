import sqlalchemy
import sqlalchemy.orm
from sqlalchemy import Column, Integer, String


engine = None
Session = None


Base = sqlalchemy.orm.declarative_base()


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
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
