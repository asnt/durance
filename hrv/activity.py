from typing import Dict, Optional

import numpy as np


def summarize(recordings: Dict) -> Dict:
    heart_rate_series = recordings.get("heart_rate")
    heart_rate: Optional[int]
    if heart_rate_series is not None:
        heart_rate = round(np.nanmedian(heart_rate_series))
    else:
        heart_rate = None

    stride_rate_series = recordings.get("cadence")
    step_rate: Optional[int]
    if stride_rate_series is not None:
        stride_rate = round(np.nanmedian(stride_rate_series))
        step_rate = 2 * stride_rate
    else:
        step_rate = None

    timestamps = recordings["timestamp"]
    time_start = timestamps[0]
    time_end = timestamps[-1]
    duration_s = time_end - time_start
    duration = duration_s.item()
    distance = recordings["distance"][-1]

    return dict(
        distance=distance,
        duration=duration,
        heart_rate=heart_rate,
        step_rate=step_rate,
    )
