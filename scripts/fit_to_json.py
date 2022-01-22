"""Convert a fit file to json."""

import argparse
import json
import pathlib
import sys

import fitparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=sys.stdout)
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

    def get_rr_array(messages):
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

    def rr_to_series(rr):
        return pd.Series(rr, name="rr")

    rr = get_rr_array(messages)
    rr_series = rr_to_series(rr)

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

    return rr_series, records_dataframe


def main():
    args = parse_args()

    rr, records = load_fit(args.input)
    arrays = {
        name: series.values.tolist()
        for name, series in records.items()
    }
    arrays["rr"] = rr.values.tolist()
    del arrays["timestamp"]

    json.dump(arrays, args.output)


if __name__ == "__main__":
    main()
