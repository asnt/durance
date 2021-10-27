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

    data = hrv.data.load_fit(args.activity_file)

    data_sport = data["sport"][0]
    name = data_sport["name"]
    sport = data_sport["sport"]
    sub_sport = data_sport["sub_sport"]

    data_file_id = data["file_id"][0]
    device_manufacturer = data_file_id["manufacturer"]
    device_model = None
    if device_manufacturer == "garmin":
        device_model = data_file_id["garmin_product"]

    datetime_start = None
    datetime_end = None
    for data_event in data["event"]:
        if not data_event["event"] == "timer":
            continue
        if data_event["event_type"] == "start":
            datetime_start = data_event["timestamp"]
        elif data_event["event_type"] == "stop_all":
            datetime_end = data_event["timestamp"]

    records = pd.DataFrame.from_records(data["record"])
    heartrate = records["heart_rate"].values
    heartrate_mean = int(np.mean(heartrate))
    heartrate_median = int(np.median(heartrate))

    timestamps = records["timestamp"].values
    time_start = timestamps[0]
    time_end = timestamps[-1]
    duration_ns = time_end - time_start
    duration_s = np.timedelta64(duration_ns, "s")
    duration = duration_s.astype(int)
    duration = duration.item()
    distance = records["distance"].values[-1]

    data = dict(
        file_hash=app.model.hash_file(args.activity_file),

        device_manufacturer=device_manufacturer,
        device_model=device_model,

        datetime_start=datetime_start,
        datetime_end=datetime_end,

        name=name,
        sport=sport,
        sub_sport=sub_sport,

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
