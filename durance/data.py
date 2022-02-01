import collections
import datetime
import os
import pathlib
from typing import Dict, Optional, Tuple

import fitparse
import numpy as np
import pandas as pd


ACTIVITY_SCHEMA = dict(
    file_hash=None,

    name=None,
    sport=None,
    sub_sport=None,
    workout=None,

    device_manufacturer=None,
    device_model=None,

    datetime_start=None,
    datetime_end=None,

    duration=None,
    distance=None,

    heartrate_mean=None,
    heartrate_median=None,
)


def _is_hrmonitorapp_activity(path: os.PathLike) -> bool:
    path = pathlib.Path(path)
    return (path.suffix.lower() == ".csv"
            and path.name.startswith("user_hr_data_"))


def load(path: os.PathLike) -> Tuple[Dict, Dict]:
    path = pathlib.Path(path)
    if path.suffix.lower() == ".fit":
        return load_fit(path)
    elif _is_hrmonitorapp_activity(path):
        return load_hrmonitorapp(path)
    raise ValueError(f"unsupported activity file format {path}")


def _rr_remove_padding(rr_padded):
    """Get the RR signal from recording segments padded with None values."""
    rr = np.array(rr_padded)
    rr = rr.flatten()
    roi_signal = rr != None
    rr = rr[roi_signal]
    return rr.astype(float)


def load_fit(path: os.PathLike) -> Tuple[Dict, Dict]:
    fit_data = fitparse.FitFile(str(path))

    messages = fit_data.messages
    message_types = set(message.name for message in messages)

    data = collections.defaultdict(list)
    for message_type in message_types:
        messages_data = [
            {data.name: data.value for data in message}
            for message in fit_data.get_messages(message_type)
        ]
        data[message_type] = messages_data

    recordings_df = pd.DataFrame.from_records(data["record"])
    recordings = {
        column: recordings_df[column].values
        for column in recordings_df.columns
    }

    if "hrv" in data:
        rr_padded = [
            hrv_record["time"]
            for hrv_record in data["hrv"]
        ]
        rr_raw = _rr_remove_padding(rr_padded)
        recordings["rr"] = rr_raw

    def numpy_datetime64_to_timestamp_s(datetime64):
        unix_epoch = np.datetime64(0, "s")
        one_second = np.timedelta64(1, "s")
        seconds_since_epoch = (datetime64 - unix_epoch) / one_second
        return seconds_since_epoch

    if "timestamp" in recordings:
        datetimes = recordings["timestamp"]
        recordings["timestamp"] = numpy_datetime64_to_timestamp_s(datetimes)

    if "sport" in data:
        data_sport = data["sport"][0]
        name = data_sport["name"]
        sport = data_sport["sport"]
        sub_sport = data_sport["sub_sport"]
    else:
        # XXX: 'sport' record not always present.
        data_session = data["session"][0]
        # XXX: No name defined in 'session' record.
        name = data_session.get("name", None)
        sport = data_session["sport"]
        sub_sport = data_session["sub_sport"]

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

    activity_data = dict(
        device_manufacturer=device_manufacturer,
        device_model=device_model,

        datetime_start=datetime_start,
        datetime_end=datetime_end,

        name=name,
        sport=sport,
        sub_sport=sub_sport,
    )

    return activity_data, recordings


def load_fit_records(path):
    fit_data = fitparse.FitFile(str(path))
    records = [
        {data.name: data.value for data in record}
        for record in fit_data.get_messages('record')
    ]
    return records


def load_rr_from_fit(path):
    fit_data = fitparse.FitFile(str(path))
    rr = []
    records = fit_data.get_messages('hrv')
    rr_intervals_with_nones = [
        record_data.value
        for record in records
        for record_data in record
    ]
    rr = np.array(rr_intervals_with_nones)
    rr = rr.flatten()
    rr = rr[rr != None]
    return rr


def load_rr_from_csv(path):
    return np.loadtxt(path)


def load_rr(path):
    if path.suffix.lower() == ".fit":
        rr_raw = load_rr_from_fit(path)
    elif path.suffix.lower() == ".csv":
        rr_raw_ms = load_rr_from_csv(path)
        rr_raw_s = rr_raw_ms / 1000
        rr_raw = rr_raw_s
    else:
        raise ValueError("input file not supported (must be .fit or .csv)")
    return rr_raw


def _hrmonitorapp_date_time_to_datetime(date: str, time_: str) \
        -> datetime.datetime:
    assert len(date) == 8 and len(date.split("/")) == 3
    assert len(time_) == 8 and len(time_.split(":")) == 3
    mm, dd, yy = date.split("/")
    datetime_iso_string = f"20{yy}-{mm}-{dd} {time_}"
    return datetime.datetime.fromisoformat(datetime_iso_string)


def _hrmonitorapp_duration_to_seconds(duration: str) -> int:
    assert len(duration) == 8 and len(duration.split(":")) == 3
    hh, mm, ss = [int(x) for x in duration.split(":")]
    seconds = hh * 3600 + mm * 60 + ss
    return seconds


def _hrmonitorapp_parse_stats(lines) -> Dict:
    pairs = [line.split(",") for line in lines]
    pairs = [(pair[0].lower(), pair[1]) for pair in pairs]
    data = dict(pairs)

    date = data["date"]
    time_ = data["start"]
    duration = data["duration"]

    datetime_start = _hrmonitorapp_date_time_to_datetime(date, time_)
    duration = _hrmonitorapp_duration_to_seconds(duration)
    datetime_end = datetime_start + datetime.timedelta(seconds=duration)

    stats = dict(
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        duration=duration,
    )

    return stats


def _get_lines_until_blank(file_):
    lines = []
    for line in file_:
        line = line.strip()
        if not line:
            break
        lines.append(line)
    return lines


def _hrmonitorapp_parse_recordings(lines_recordings: list[str]) -> Dict:
    tokens = [line.split(",") for line in lines_recordings]

    headers = tokens[0]
    headers = [header.lower() for header in headers]

    renames = dict(sec="timestamp", hr_bpm="heart_rate")
    headers = [renames.get(header, header) for header in headers]

    values = np.array(tokens[1:], dtype=float)
    values = values.T

    recordings = {
        header: series
        for header, series in zip(headers, values)
    }
    return recordings


def load_hrmonitorapp(path: os.PathLike) -> Tuple[Dict, Dict]:
    data = dict(
        file_hash=None,

        name=None,
        sport=None,
        sub_sport=None,
        workout=None,

        device_manufacturer=None,
        device_model=None,

        datetime_start=None,
        datetime_end=None,

        duration=None,
        distance=None,

        heartrate_mean=None,
        heartrate_median=None,
    )

    with open(path, "r") as file_:
        for line in file_:
            line = line.strip()
            if line == "{Statistics}":
                lines_stats = _get_lines_until_blank(file_)
                stats = _hrmonitorapp_parse_stats(lines_stats)
                data.update(stats)
            elif line == "{History}":
                lines_recordings = _get_lines_until_blank(file_)
                recordings = _hrmonitorapp_parse_recordings(lines_recordings)

    return data, recordings
