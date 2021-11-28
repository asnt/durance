import argparse
import pathlib
import os

import hrv.activity
import hrv.data
import app.model


def init_db(args) -> None:
    del args
    engine = app.model.make_engine()
    app.model.create(engine)


def import_activities(args) -> None:
    print(f"importing {len(args.files)} activities")
    for path in args.files:
        import_activity(path)


def import_activity(path: os.PathLike) -> None:
    path = pathlib.Path(path)
    print(f"importing {path}")

    if app.model.has_activity(path):
        print("activity already imported")
        # return

    activity_data, recordings_data = hrv.data.load(path)
    activity_data["file_hash"] = app.model.hash_file(path)
    summary_data = hrv.activity.summarize(recordings_data)

    _ = app.model.make_engine()
    session = app.model.make_session()

    activity = app.model.Activity(**activity_data)
    session.add(activity)
    session.commit()

    # Note: Different commit to have an initialized activity id.
    summary = app.model.Summary(**summary_data, activity_id=activity.id)
    session.add(summary)
    session.commit()

    recordings = [
        app.model.Recording(
            activity=activity,
            name=name,
            array=data,
        )
        for name, data in recordings_data.items()
    ]
    for recording in recordings:
        session.add(recording)
    session.commit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=pathlib.Path, default="activities.db")

    subparsers = parser.add_subparsers()

    parser_init = subparsers.add_parser("init")
    parser_init.set_defaults(func=init_db)

    parser_import = subparsers.add_parser("import")
    parser_import.set_defaults(func=import_activities)
    parser_import.add_argument("files",
                               nargs="+",
                               type=pathlib.Path,
                               help="FIT activity file(s).")

    return parser.parse_args()


def main():
    args = parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
