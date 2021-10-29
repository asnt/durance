import argparse
import pathlib
import os

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
        return

    activity_data, _ = hrv.data.load(path)
    activity_data["file_hash"] = app.model.hash_file(path)

    activity = app.model.Activity(**activity_data)

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
