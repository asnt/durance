import argparse
import os
import pathlib
import sqlite3


sql_init_tables = """
create table activities (
    id integer primary key,
    type text,
    duration real,
    distance real,
    heartrate_mean real,
    heartrate_median real
);
"""

# TODO: Add adapter/converter to store/read numpy array into/from sqlite.
# https://stackoverflow.com/a/18622264


def db_connect(path_like: os.PathLike) -> sqlite3.Connection:
    return sqlite3.connect(path_like)


def db_init(db: sqlite3.Connection, args) -> None:
    cursor = db.cursor()
    cursor.executescript(sql_init_tables)
    db.commit()


def import_activity(db: sqlite3.Connection, args) -> None:
    print(f"importing {args.activity_file}")
    import numpy as np
    import pandas as pd
    import hrv.data
    records = hrv.data.load_fit_records(args.activity_file)
    records = pd.DataFrame.from_records(records)
    heartrate = records["heart_rate"].values
    heartrate_mean = np.mean(heartrate)
    heartrate_median = np.median(heartrate)

    timestamps = records["timestamp"].values
    time_start = timestamps[0]
    time_end = timestamps[-1]
    duration_ns = time_end - time_start
    duration_s = np.timedelta64(duration_ns, "s")
    duration = duration_s.astype(int)
    duration = duration.item()
    distance = records["distance"].values[-1]

    data = dict(
        type="undefined",
        duration=duration,
        distance=distance,
        heartrate_mean=heartrate_mean,
        heartrate_median=heartrate_median,
    )

    cursor = db.cursor()
    names = ",".join(data)
    placeholders = ",".join("?" * len(data))
    query = f"insert into activities ({names}) values ({placeholders})"
    cursor.execute(query, list(data.values()))
    db.commit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=pathlib.Path, default="activities.db")

    subparsers = parser.add_subparsers()

    parser_init = subparsers.add_parser("init")
    parser_init.set_defaults(func=db_init)

    parser_import = subparsers.add_parser("import")
    parser_import.set_defaults(func=import_activity)
    parser_import.add_argument("activity_file",
                               type=pathlib.Path,
                               help="FIT activity file.")

    return parser.parse_args()


def main():
    args = parse_args()

    db = sqlite3.connect(args.db)

    args.func(db, args)


if __name__ == "__main__":
    main()
