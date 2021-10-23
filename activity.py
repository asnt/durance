import argparse
import os
import pathlib
import sqlite3


sql_init_tables = """
create table activities (
    id integer primary key,
    data text,
    type text,
    duration real,
    length real
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
    print(records)
    print(records.columns)
    heartrate = records["heart_rate"].values
    heartrate_mean = np.mean(heartrate)
    heartrate_median = np.median(heartrate)
    heartrate_min = np.min(heartrate)
    heartrate_max = np.max(heartrate)
    print("mean(hr) =", heartrate_mean)
    print("median(hr) =", heartrate_median)
    print("min(hr) =", heartrate_min)
    print("max(hr) =", heartrate_max)
    pass


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
