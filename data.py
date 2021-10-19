import fitparse
import numpy as np


def load_fit_records(path):
    fit_data = fitparse.FitFile(str(args.input))
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
