"""Display a summary of the contents of a .fit file to stdout."""

import argparse
import pathlib

import fitparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    return parser.parse_args()


def load_fit(path):
    fit_data = fitparse.FitFile(str(path))
    messages = fit_data.messages

    # DataMessage types:
    # - record (#21)
    # - event (#21)
    # - device_info (#23)
    # - hrv (#78)
    # - unknown (#140)
    # - ...

    def list_message_types(messages):
        import collections
        message_types = collections.defaultdict(int)
        for message in messages:
            key = (message.mesg_num, message.name)
            message_types[key] += 1
        return dict(message_types)

    import pprint
    print("message types:")
    message_types = list_message_types(messages)
    pprint.pprint(message_types)

    def get_rr_series(messages):
        records = fit_data.get_messages(name="hrv")
        rr_intervals_with_nones = [
            record_data.value
            for record in records
            for record_data in record
        ]
        rr = np.array(rr_intervals_with_nones)
        rr = rr.flatten()
        rr = rr[rr != None]
        return rr

    def rr_to_dataframe(rr):
        return pd.DataFrame(dict(rr=rr))

    rr = get_rr_series(messages)
    rr_dataframe = rr_to_dataframe(rr)

    def get_records(messages):
        records_messages = fit_data.get_messages(name="record")
        records = [
            {data.name: data.value for data in record}
            for record in records_messages
        ]
        return records

    def records_to_dataframe(records):
        return pd.DataFrame.from_records(records)

    def start_datetime_from_dataframe(dataframe):
        start_timestamp = dataframe["timestamp"][0]
        start_datetime = start_timestamp.to_pydatetime()
        return start_datetime

    records = get_records(messages)
    records_dataframe = records_to_dataframe(records)

    return rr_dataframe, records_dataframe


def main():
    args = parse_args()

    rr, records = load_fit(args.input)

    print(rr)
    print(records)


if __name__ == "__main__":
    main()
