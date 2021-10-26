import argparse
import pathlib

import app.model


def init_db(args) -> None:
    del args
    engine = app.model.make_engine()
    app.model.create(engine)


def import_activity(args) -> None:
    print(f"importing {args.activity_file}")

    import numpy as np
    import pandas as pd
    import hrv.data

    if app.model.has_activity(args.activity_file):
        print("activity already imported")
        return

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
        name="unnamed",
        file_hash=app.model.hash_file(args.activity_file),
        type="undefined",
        duration=duration,
        distance=distance,
        heartrate_mean=heartrate_mean,
        heartrate_median=heartrate_median,
    )

    activity = app.model.Activity(**data)
    print(activity)

    _ = app.model.make_engine()
    session = app.model.make_session()
    session.add(activity)
    session.commit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=pathlib.Path, default="activities.db")

    subparsers = parser.add_subparsers()

    parser_init = subparsers.add_parser("init")
    parser_init.set_defaults(func=init_db)

    parser_import = subparsers.add_parser("import")
    parser_import.set_defaults(func=import_activity)
    parser_import.add_argument("activity_file",
                               type=pathlib.Path,
                               help="FIT activity file.")

    return parser.parse_args()


def main():
    args = parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
