"""Display FIT record and hrv messages side by side to check alignment.

- HRV messages do not have an explicit time stamp.
- But, the cumulated durations of the RR intervals in the HRV signal should
  give relative durations of HRV samples.
- Otherwise, HRV messages are interleaved with record messages. An association
  between the two types can be deduced, assuming the order of the messages
  matches the time ordering.
"""

import argparse
import pathlib

import fitparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    return parser.parse_args()


def extract_rr_intervals(
    messages: list[fitparse.records.DataMessage],
) -> (np.ndarray, np.ndarray):
    """
    Returns
    -------
    rr_intervals:
        RR interval durations.
    indices:
        Record index corresponding to each RR interval.

    """
    rr_intervals_with_nones = [
        record_data.value
        for record in messages
        for record_data in record
    ]
    rr = np.array(rr_intervals_with_nones)
    mask_defined = rr != None
    indices_defined = np.nonzero(mask_defined)
    rr = rr[indices_defined]
    indices_record = indices_defined[0]
    return rr, indices_record


def main():
    args = parse_args()

    fit_data: fitparse.FitFile = fitparse.FitFile(str(args.input))
    messages: list = fit_data.messages
    messages_record_and_hrv: list = [
        message
        for message in messages
        if message.name in ["record", "hrv"]
    ]

    index_last_record: int = -1
    index_hrv_to_record: list = []
    for message in messages_record_and_hrv:
        if message.name == "record":
            index_last_record += 1
        elif message.name == "hrv":
            index_hrv_to_record.append(index_last_record)

    messages_record = [
        message
        for message in messages_record_and_hrv
        if message.name == "record"
    ]
    messages_hrv = [
        message
        for message in messages_record_and_hrv
        if message.name == "hrv"
    ]
    rr, rr_record_indices = extract_rr_intervals(messages_hrv)
    _, index_hrv_to_rr_packet = np.unique(rr_record_indices, return_index=True)
    rr_cumulated = np.cumsum(rr)
    print(f"#messages record = {len(messages_record)}")
    print(f"#messages hrv = {len(messages_hrv)}")
    print(f"#RR intervals = {len(rr)}")

    start_time = messages_record[0].get("timestamp").as_dict()["value"]
    print(f"record start time = {start_time}")
    print("Table header:")
    print("\n ".join([
        " index_hrv",
        "index_record",
        "record_timestamp",
        "record_total_s",
        "rr_total_s",
        "hrv_times",
    ]))
    index_rr = 0
    for index_hrv, hrv in enumerate(messages_hrv):
        index_record = index_hrv_to_record[index_hrv]
        record = messages_record[index_record]
        record_timestamp = record.get("timestamp").as_dict()["value"]
        record_timedelta = record_timestamp - start_time
        record_relative_time_s = record_timedelta.total_seconds()
        hrv_times = hrv.get("time").as_dict()["value"]
        index_rr = index_hrv_to_rr_packet[index_hrv]
        rr_cumulated_ = rr_cumulated[index_rr]
        log_message = f"{index_hrv:04d}"
        log_message += f"  {index_record:04d}"
        log_message += f"  {record_timestamp}"
        log_message += f"  {record_relative_time_s:7.2f}"
        log_message += f"  {rr_cumulated_:7.2f}"
        log_message += f"  {hrv_times}"
        print(log_message)


if __name__ == "__main__":
    main()
